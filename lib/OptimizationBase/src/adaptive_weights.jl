# Objective functor produced by `weighted_sum`: evaluates
# `dot(weights, mof.f(u, p))` where `mof` is the source
# `MultiObjectiveOptimizationFunction`. The `weights` vector is shared with the
# adaptive-weight rules, which mutate it in place during a solve.
struct WeightedSumObjective{F, W}
    mof::F
    weights::W
end

(o::WeightedSumObjective)(u, p) = dot(o.weights, o.mof.f(u, p))

_jac_buffer(u::Number, n) = Matrix{typeof(u)}(undef, n, 1)
_jac_buffer(u::AbstractVector, n) = similar(u, n, length(u))
function _jac_buffer(u, n)
    throw(
        ArgumentError(
            "cannot build a per-objective Jacobian buffer for decision variables " *
                "of type $(typeof(u)); pass a vector-valued `u0`."
        )
    )
end

function _weighted_grad(mof, w, ::Val{true})
    return function (G, u, p)
        J = _jac_buffer(u, length(w))
        mof.jac(J, u, p)
        return mul!(G, transpose(J), w)
    end
end
function _weighted_grad(mof, w, ::Val{false})
    return (u, p) -> transpose(mof.jac(u, p)) * w
end

"""
    weighted_sum(f::MultiObjectiveOptimizationFunction, weights; adtype = f.adtype)

Scalarize the multi-objective `f` into a single-objective `OptimizationFunction`
evaluating `dot(weights, f(u, p))`. `weights` is copied into an internal vector
that adaptive-weight rules mutate during a solve; read the adapted weights back
from the rule's `weights` field.

When `f.jac` is provided, the returned function carries a `grad` implementing
`∇(weights ⋅ f) = f.jacᵀ * weights` under the current weights. Otherwise
derivatives are left to `adtype`, which defaults to `f.adtype`. Constraint
callbacks (`cons`, `cons_j`, `cons_h`, `cons_jvp`, `cons_vjp`) and their
prototypes and color vectors are carried over unchanged.

    weighted_sum(prob::OptimizationProblem; weights = nothing, adtype = prob.f.adtype)

Scalarize `prob`, whose `f` must be a `MultiObjectiveOptimizationFunction`, and
return a new `OptimizationProblem` of the same shape (`u0`, `p`, bounds,
constraint limits, and sense are preserved). `weights` defaults to `ones` with
one entry per objective, counted by evaluating `prob.f.f(prob.u0, prob.p)`.

The adaptive-weight rules assume minimization of the weighted sum (they ascend
weights against objective values); `sense = MaxSense` is preserved on the
problem but is not handled specially by the rules.

Pass the returned problem to an adaptive-weight rule ([`GradientScale`](@ref),
[`MiniMax`](@ref), [`SoftAdapt`](@ref), [`ReLoBRaLo`](@ref)) to build the
`solve` callback that updates the weights during the solve.

# Example

```julia
mof = MultiObjectiveOptimizationFunction(
    (u, p) -> [u[1]^2, (u[1] - 2)^2], AutoForwardDiff()
)
prob = OptimizationProblem(mof, [1.0])
sprob = weighted_sum(prob)
cb = ReLoBRaLo(sprob; every = 10)
sol = solve(sprob, opt; callback = cb, save_best = false)
```
"""
function weighted_sum(
        f::SciMLBase.MultiObjectiveOptimizationFunction{iip}, weights;
        adtype = f.adtype
    ) where {iip}
    w = float.(collect(weights))
    obj = WeightedSumObjective(f, w)
    grad = f.jac === nothing ? nothing : _weighted_grad(f, w, Val(iip))
    return OptimizationFunction{iip}(
        obj, adtype;
        grad = grad,
        cons = f.cons, cons_j = f.cons_j, cons_jvp = f.cons_jvp,
        cons_vjp = f.cons_vjp, cons_h = f.cons_h,
        cons_jac_prototype = f.cons_jac_prototype,
        cons_hess_prototype = f.cons_hess_prototype,
        cons_jac_colorvec = f.cons_jac_colorvec,
        cons_hess_colorvec = f.cons_hess_colorvec,
        observed = f.observed
    )
end

function weighted_sum(
        prob::SciMLBase.OptimizationProblem; weights = nothing, adtype = prob.f.adtype
    )
    return _weighted_sum_prob(prob, weights; adtype)
end

function _weighted_sum_prob(prob, weights; adtype)
    f = prob.f
    f isa SciMLBase.MultiObjectiveOptimizationFunction || throw(
        ArgumentError(
            "`weighted_sum` scalarizes multi-objective problems: `prob.f` must be a " *
                "`MultiObjectiveOptimizationFunction`, got $(typeof(f))."
        )
    )
    values = f.f(prob.u0, prob.p)
    values isa AbstractVector || throw(
        ArgumentError(
            "`weighted_sum` expects `f(u, p)` to return a vector of objectives, " *
                "got $(typeof(values))."
        )
    )
    n = length(values)
    w = weights === nothing ? ones(n) : weights
    length(w) == n || throw(
        DimensionMismatch(
            "`weights` has $(length(w)) entries but the objective returns $n values."
        )
    )
    return remake(prob; f = weighted_sum(f, w; adtype))
end

"""
    AbstractAdaptiveWeightRule

Supertype for `solve` callbacks that adapt the weights of a
[`weighted_sum`](@ref) scalarization every `every` iterations.

Every rule is constructed from the scalarized problem returned by `weighted_sum`
and passed as the `callback` keyword of `solve`. Rules read `state.iter`,
`state.u`, and `state.p` and always return `false`.

Weight updates fire when `state.iter` is a multiple of `every`. Solvers that
never advance `state.iter` (it stays at the default `0`) therefore update once
at the first callback and then skip later calls as same-iteration repeats. Prefer
optimizers that set a strictly increasing `state.iter` each callback — for
example `OptimizationOptimisers` algorithms. Separately, some solver callback
states leave `p` as `nothing`; the rules then evaluate `objectives(u, nothing)`.

Prefer `save_best = false` with `OptimizationOptimisers` algorithms: the default
`save_best = true` stores the iterate with the lowest *weighted* objective, but
objective values under changing weights are not comparable, and on the final
iteration `save_best` reverts `θ` and re-invokes the callback at the same
`state.iter` (an update the rules skip).

Reusing a rule across solves resets the same-iteration guard when `state.iter`
goes strictly backwards relative to the previous call. A second solve that
starts at exactly the same iteration the previous solve ended on does not
trigger that reset; construct a fresh rule for that edge case.
"""
abstract type AbstractAdaptiveWeightRule end

function _check_update_period(every)
    every > 0 || throw(ArgumentError("`every` must be a positive integer."))
    return Int(every)
end

function _adaptive_context(prob)
    of = prob.f
    obj = of isa SciMLBase.OptimizationFunction ? of.f : nothing
    obj isa WeightedSumObjective || throw(
        ArgumentError(
            "adaptive-weight rules apply to a problem scalarized by `weighted_sum`; " *
                "build it with `weighted_sum(prob)` first and pass the returned problem."
        )
    )
    return (
        objectives = obj.mof.f, jac = obj.mof.jac,
        iip_jac = isinplace(obj.mof), adtype = of.adtype, weights = obj.weights,
    )
end

# Return `true` when `state.iter` is due for a weight update. Skips a second call
# at the same iteration (the OptimizationOptimisers `save_best` finalization
# path), and resets the guard when `state.iter` goes backwards (the callback
# object was reused for a new solve). A new solve that starts at exactly the
# previous solve's last iteration does not reset; build a fresh rule then.
function _due(rule, state)
    iter = state.iter
    if iter < rule.last_seen
        rule.last_updated = -1
    end
    rule.last_seen = iter
    iter % rule.every == 0 || return false
    rule.last_updated == iter && return false
    rule.last_updated = iter
    return true
end

function _softmax_weights(scores)
    shifted = scores .- maximum(scores)
    values = exp.(shifted)
    return length(values) .* values ./ sum(values)
end

_meanabs(g) = sum(abs, g) / length(g)

function _objective_jacobian(objectives, jac, iip_jac, adtype, u, p, nobj)
    if jac !== nothing
        if iip_jac
            J = _jac_buffer(u, nobj)
            jac(J, u, p)
            return J
        else
            return jac(u, p)
        end
    else
        return jacobian(u′ -> objectives(u′, p), adtype, u)
    end
end

"""
    GradientScale(prob; every = 1, inertia = 0.9, epsilon = 1e-11)

Return a `solve` callback that updates the [`weighted_sum`](@ref) weights of `prob`
from their per-objective gradient magnitudes, following Wang, Teng, and Perdikaris
(2020), [Understanding and mitigating gradient pathologies in physics-informed neural
networks](https://arxiv.org/abs/2001.04536).

At each update, per-objective gradients `∇Lᵢ(u)` are collected from the problem's
Jacobian (the `jac` of the `MultiObjectiveOptimizationFunction` when provided,
otherwise differentiated through `adtype`), and the largest maximum absolute
gradient is divided by each objective's mean absolute gradient:

```math
\\hat{w}_i = \\frac{\\max_j \\max_u |\\nabla L_j|}{\\overline{|\\nabla L_i|} + \\epsilon}
```

`inertia` exponentially averages the proposed weights,
`w ← inertia * w + (1 - inertia) * ŵ`. Per-objective gradients require either a
`jac` in the `MultiObjectiveOptimizationFunction` or a non-`NoAD` `adtype`.
Prefer `save_best = false` when solving (see [`AbstractAdaptiveWeightRule`](@ref)).
"""
mutable struct GradientScale{F, J, A, W, T} <: AbstractAdaptiveWeightRule
    objectives::F
    jac::J
    iip_jac::Bool
    adtype::A
    weights::W
    every::Int
    inertia::T
    epsilon::T
    last_updated::Int
    last_seen::Int
end

function GradientScale(prob; every = 1, inertia = 0.9, epsilon = 1.0e-11)
    0 <= inertia <= 1 || throw(ArgumentError("`inertia` must be between zero and one."))
    ctx = _adaptive_context(prob)
    ctx.jac === nothing && ctx.adtype isa SciMLBase.NoAD && throw(
        ArgumentError(
            "`GradientScale` needs per-objective gradients: pass `jac` to the " *
                "`MultiObjectiveOptimizationFunction` or scalarize with a non-`NoAD` `adtype`."
        )
    )
    return GradientScale(
        ctx.objectives, ctx.jac, ctx.iip_jac, ctx.adtype, ctx.weights,
        _check_update_period(every), inertia, epsilon, -1, -1
    )
end

function (rule::GradientScale)(state, loss)
    _due(rule, state) || return false
    state.u === nothing && return false
    J = _objective_jacobian(
        rule.objectives, rule.jac, rule.iip_jac, rule.adtype,
        state.u, state.p, length(rule.weights)
    )
    scale = maximum(abs, J)
    iszero(scale) && return false
    proposed = [scale / (_meanabs(g) + rule.epsilon) for g in eachrow(J)]
    @. rule.weights = rule.inertia * rule.weights + (1 - rule.inertia) * proposed
    return false
end

"""
    MiniMax(prob; every = 1, η = 0.5)

Return a `solve` callback that ascends the [`weighted_sum`](@ref) weights of `prob`
by a plain gradient-ascent step on the objective values, following McClenny and
Braga-Neto (2020), [Self-Adaptive PINNs](https://arxiv.org/abs/2009.04544). The
weighted-sum objective `Σ wᵢ Lᵢ` is linear in the weights, so

```math
w ← w + η \\, L(u)
```

where `L(u)` is the vector of objective values at the current iterate and `η` is
the learning rate. Prefer `save_best = false` when solving (see
[`AbstractAdaptiveWeightRule`](@ref)).
"""
mutable struct MiniMax{F, W, T} <: AbstractAdaptiveWeightRule
    objectives::F
    weights::W
    every::Int
    η::T
    last_updated::Int
    last_seen::Int
end

function MiniMax(prob; every = 1, η = 0.5)
    η > 0 || throw(ArgumentError("`η` must be positive."))
    ctx = _adaptive_context(prob)
    return MiniMax(
        ctx.objectives, ctx.weights, _check_update_period(every), η, -1, -1
    )
end

function (rule::MiniMax)(state, loss)
    _due(rule, state) || return false
    state.u === nothing && return false
    values = collect(rule.objectives(state.u, state.p))
    @. rule.weights = rule.weights + rule.η * values
    return false
end

"""
    SoftAdapt(prob; every = 1, α = 0.1, epsilon = 1e-8)

Return a `solve` callback that weights the objectives of `prob` by a softmax of
their relative loss changes, following Heydari, Thompson, and Mehmood (2019),
[SoftAdapt](https://arxiv.org/abs/1912.12355). On the first due iteration only the
objective values are recorded; from the second on,

```math
w_i = N \\, \\mathrm{softmax}_i\\!\\left(\\alpha \\frac{L_i(u_t) - L_i(u_{t-1})}{L_i(u_{t-1}) + \\epsilon}\\right)
```

where `N` is the number of objectives. Prefer `save_best = false` when solving (see
[`AbstractAdaptiveWeightRule`](@ref)); with SoftAdapt a `save_best` double-fire at
the final iteration can otherwise reset the stored relative-rate state.
"""
mutable struct SoftAdapt{F, W, T} <: AbstractAdaptiveWeightRule
    objectives::F
    weights::W
    every::Int
    α::T
    epsilon::T
    previous::Any
    last_updated::Int
    last_seen::Int
end

function SoftAdapt(prob; every = 1, α = 0.1, epsilon = 1.0e-8)
    ctx = _adaptive_context(prob)
    return SoftAdapt(
        ctx.objectives, ctx.weights, _check_update_period(every), α, epsilon,
        nothing, -1, -1
    )
end

function (rule::SoftAdapt)(state, loss)
    _due(rule, state) || return false
    state.u === nothing && return false
    values = collect(rule.objectives(state.u, state.p))
    if rule.previous !== nothing
        scores = rule.α .* (values .- rule.previous) ./ (rule.previous .+ rule.epsilon)
        rule.weights .= _softmax_weights(scores)
    end
    rule.previous = values
    return false
end

"""
    ReLoBRaLo(prob; every = 1, α = 0.99, β = 0.9, temperature = 1.0,
        epsilon = 1e-8, rng = Random.default_rng())

Return a `solve` callback implementing Relative Loss Balancing with Random Lookback
on the [`weighted_sum`](@ref) weights of `prob`, following Bischof and Kraus (2021),
[Multi-Objective Loss Balancing for Physics-Informed Deep
Learning](https://arxiv.org/abs/2110.09813). `α` controls the published moving
average, `β` is the probability of carrying the previous scalings forward, and
`temperature` is the paper's `𝒯`: higher values flatten the softmax toward uniform
weights. Prefer `save_best = false` when solving (see
[`AbstractAdaptiveWeightRule`](@ref)).
"""
mutable struct ReLoBRaLo{F, W, T, R} <: AbstractAdaptiveWeightRule
    objectives::F
    weights::W
    every::Int
    α::T
    β::T
    temperature::T
    epsilon::T
    rng::R
    initial_losses::Any
    previous_losses::Any
    previous_weights::Any
    last_updated::Int
    last_seen::Int
end

function ReLoBRaLo(
        prob; every = 1, α = 0.99, β = 0.9, temperature = 1.0,
        epsilon = 1.0e-8, rng = Random.default_rng()
    )
    (0 <= α <= 1 && 0 <= β <= 1) || throw(
        ArgumentError("`α` and `β` must be between zero and one.")
    )
    temperature > 0 || throw(ArgumentError("`temperature` must be positive."))
    ctx = _adaptive_context(prob)
    return ReLoBRaLo(
        ctx.objectives, ctx.weights, _check_update_period(every), α, β,
        temperature, epsilon, rng, nothing, nothing, nothing, -1, -1
    )
end

function (rule::ReLoBRaLo)(state, loss)
    _due(rule, state) || return false
    state.u === nothing && return false
    values = collect(rule.objectives(state.u, state.p))
    if rule.initial_losses === nothing
        rule.initial_losses = values
        rule.previous_losses = values
        rule.previous_weights = copy(rule.weights)
    else
        rho = rand(rule.rng) < rule.β
        initial_balance = _softmax_weights(
            values ./ (rule.temperature .* (rule.initial_losses .+ rule.epsilon))
        )
        previous_balance = _softmax_weights(
            values ./ (rule.temperature .* (rule.previous_losses .+ rule.epsilon))
        )
        updated = rule.α .* (
            rho .* rule.previous_weights .+
                (1 - rho) .* initial_balance
        ) .+ (1 - rule.α) .* previous_balance
        rule.weights .= updated
        rule.previous_losses = values
        rule.previous_weights = copy(updated)
    end
    return false
end
