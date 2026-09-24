module OptimizationManopt

using Manopt
import OptimizationBase
using SciMLLogging: @SciMLMessage
using ManifoldsBase, ManifoldDiff
import SciMLBase
using SciMLBase: ReturnCode
using Dates: Millisecond

"""
    abstract type AbstractManoptOptimizer end

A Manopt solver without things specified by a call to `solve` (stopping criteria) and
internal state.
"""
abstract type AbstractManoptOptimizer end

SciMLBase.has_init(opt::AbstractManoptOptimizer) = true
SciMLBase.allowscallback(opt::AbstractManoptOptimizer) = true
OptimizationBase.supports_sense(::AbstractManoptOptimizer) = true

function __map_optimizer_args!(
        cache::OptimizationBase.OptimizationCache,
        opt::AbstractManoptOptimizer,
        manifold;
        callback = nothing,
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        stopping_criterion::Union{Manopt.StoppingCriterion, Nothing} = nothing,
        kwargs...
    )
    criteria = Manopt.StoppingCriterion[]

    if !isnothing(maxiters)
        push!(criteria, Manopt.StopAfterIteration(maxiters))
    end

    if !isnothing(maxtime)
        push!(criteria, Manopt.StopAfter(Millisecond(round(Int, maxtime * 1000))))
    end

    if !isnothing(stopping_criterion)
        # A Manopt `stopping_criterion` passed by the user is combined with the other
        # explicitly requested criteria (`maxiters`/`maxtime`/`abstol`) rather than
        # silently replaced by them, and it suppresses the convergence fallback below.
        push!(criteria, stopping_criterion)
    end

    if !isnothing(abstol)
        push!(criteria, _default_convergence_criterion(opt, manifold, abstol))
    elseif isnothing(stopping_criterion) && (!isnothing(maxiters) || !isnothing(maxtime))
        # Without this, `criteria` above would contain *only* `StopAfterIteration`/
        # `StopAfter`, which never `indicates_convergence` (Manopt.jl semantics), so
        # `Manopt.has_converged` could never be true and the run would always report
        # `ReturnCode.MaxIters`/`MaxTime` below, no matter how well it actually converged.
        # Re-adding the convergence part of the solver's own Manopt default keeps the run
        # behaving as if `maxiters`/`maxtime` merely tightened the default criterion
        # instead of discarding it; solvers whose Manopt default cannot indicate
        # convergence opt out via `_manopt_default_convergence_criterion` returning
        # `nothing` (see the comment there).
        fallback = _manopt_default_convergence_criterion(opt, manifold)
        if !isnothing(fallback)
            push!(criteria, fallback)
        end
    end

    if !isnothing(reltol)
        @SciMLMessage(
            lazy"common reltol is currently not used by $(typeof(opt).super)",
            cache.verbose, :unsupported_kwargs
        )
    end

    solver_kwargs = (; kwargs...)
    if !isempty(criteria)
        solver_kwargs = (; solver_kwargs..., stopping_criterion = criteria)
    end
    return solver_kwargs
end

## gradient descent
"""
    GradientDescentOptimizer()

Manopt gradient descent optimizer for manifold-valued Optimization.jl problems.
"""
struct GradientDescentOptimizer <: AbstractManoptOptimizer end

function call_manopt_optimizer(
        M::ManifoldsBase.AbstractManifold, opt::GradientDescentOptimizer,
        loss,
        gradF,
        x0;
        hessF = nothing, # ignore that keyword for this solver
        kwargs...
    )
    opts = Manopt.gradient_descent(
        M,
        loss,
        gradF,
        x0;
        return_state = true, # return the (full, decorated) solver state
        kwargs...
    )
    minimizer = Manopt.get_solver_result(opts)
    return (; minimizer = minimizer, minimum = loss(M, minimizer), options = opts)
end

## Nelder-Mead
"""
    NelderMeadOptimizer()

Manopt Nelder-Mead optimizer for derivative-free manifold optimization.
"""
struct NelderMeadOptimizer <: AbstractManoptOptimizer end

function call_manopt_optimizer(
        M::ManifoldsBase.AbstractManifold, opt::NelderMeadOptimizer,
        loss,
        gradF,
        x0;
        hessF = nothing, # ignore that keyword for this solver
        kwargs...
    )
    opts = NelderMead(M, loss; return_state = true, kwargs...)
    minimizer = Manopt.get_solver_result(opts)
    return (; minimizer = minimizer, minimum = loss(M, minimizer), options = opts)
end

## conjugate gradient descent
"""
    ConjugateGradientDescentOptimizer()

Manopt conjugate gradient descent optimizer for manifold-valued problems.
"""
struct ConjugateGradientDescentOptimizer <: AbstractManoptOptimizer end

function call_manopt_optimizer(
        M::ManifoldsBase.AbstractManifold,
        opt::ConjugateGradientDescentOptimizer,
        loss,
        gradF,
        x0;
        hessF = nothing, # ignore that keyword for this solver
        kwargs...
    )
    opts = Manopt.conjugate_gradient_descent(
        M,
        loss,
        gradF,
        x0;
        return_state = true,
        kwargs...
    )
    minimizer = Manopt.get_solver_result(opts)
    return (; minimizer = minimizer, minimum = loss(M, minimizer), options = opts)
end

## particle swarm
"""
    ParticleSwarmOptimizer()

Manopt particle swarm optimizer for derivative-free manifold optimization.
"""
struct ParticleSwarmOptimizer <: AbstractManoptOptimizer end

function call_manopt_optimizer(
        M::ManifoldsBase.AbstractManifold,
        opt::ParticleSwarmOptimizer,
        loss,
        gradF,
        x0;
        hessF = nothing, # ignore that keyword for this solver
        population_size::Int = 100,
        kwargs...
    )
    swarm = [x0, [rand(M) for _ in 1:(population_size - 1)]...]
    opts = particle_swarm(M, loss, swarm; return_state = true, kwargs...)
    minimizer = Manopt.get_solver_result(opts)
    return (; minimizer = minimizer, minimum = loss(M, minimizer), options = opts)
end

## quasi Newton

"""
    QuasiNewtonOptimizer()

Manopt quasi-Newton optimizer for manifold-valued problems.
"""
struct QuasiNewtonOptimizer <: AbstractManoptOptimizer end

function call_manopt_optimizer(
        M::Manopt.AbstractManifold,
        opt::QuasiNewtonOptimizer,
        loss,
        gradF,
        x0;
        hessF = nothing, # ignore that keyword for this solver
        kwargs...
    )
    opts = quasi_Newton(M, loss, gradF, x0; return_state = true, kwargs...)
    minimizer = Manopt.get_solver_result(opts)
    return (; minimizer = minimizer, minimum = loss(M, minimizer), options = opts)
end

"""
    CMAESOptimizer()

Manopt covariance matrix adaptation evolution strategy optimizer.
"""
struct CMAESOptimizer <: AbstractManoptOptimizer end

function call_manopt_optimizer(
        M::ManifoldsBase.AbstractManifold,
        opt::CMAESOptimizer,
        loss,
        gradF,
        x0;
        hessF = nothing, # ignore that keyword for this solver
        kwargs...
    )
    opt = cma_es(M, loss, x0; return_state = true, kwargs...)
    minimizer = Manopt.get_solver_result(opt)
    return (; minimizer = minimizer, minimum = loss(M, minimizer), options = opt)
end

"""
    ConvexBundleOptimizer()

Manopt convex bundle method optimizer for manifold-valued problems.
"""
struct ConvexBundleOptimizer <: AbstractManoptOptimizer end

function call_manopt_optimizer(
        M::ManifoldsBase.AbstractManifold,
        opt::ConvexBundleOptimizer,
        loss,
        gradF,
        x0;
        hessF = nothing, # ignore that keyword for this solver
        kwargs...
    )
    opt = convex_bundle_method(M, loss, gradF, x0; return_state = true, kwargs...)
    minimizer = Manopt.get_solver_result(opt)
    return (; minimizer = minimizer, minimum = loss(M, minimizer), options = opt)
end

"""
    AdaptiveRegularizationCubicOptimizer()

Manopt adaptive regularization with cubics optimizer. Uses the Hessian when the
`OptimizationFunction` supplies one.
"""
struct AdaptiveRegularizationCubicOptimizer <: AbstractManoptOptimizer end

function call_manopt_optimizer(
        M::ManifoldsBase.AbstractManifold,
        opt::AdaptiveRegularizationCubicOptimizer,
        loss,
        gradF,
        x0;
        hessF = nothing,
        kwargs...
    )
    opt = if isnothing(hessF)
        adaptive_regularization_with_cubics(
            M, loss, gradF, x0; return_state = true, kwargs...
        )
    else
        adaptive_regularization_with_cubics(
            M, loss, gradF, hessF, x0; return_state = true, kwargs...
        )
    end
    minimizer = Manopt.get_solver_result(opt)
    return (; minimizer = minimizer, minimum = loss(M, minimizer), options = opt)
end

"""
    TrustRegionsOptimizer()

Manopt Riemannian trust-regions optimizer. Uses the Hessian when the
`OptimizationFunction` supplies one.
"""
struct TrustRegionsOptimizer <: AbstractManoptOptimizer end

function call_manopt_optimizer(
        M::ManifoldsBase.AbstractManifold,
        opt::TrustRegionsOptimizer,
        loss,
        gradF,
        x0;
        hessF = nothing,
        kwargs...
    )
    opt = if isnothing(hessF)
        trust_regions(M, loss, gradF, x0; return_state = true, kwargs...)
    else
        trust_regions(M, loss, gradF, hessF, x0; return_state = true, kwargs...)
    end
    minimizer = Manopt.get_solver_result(opt)
    return (; minimizer = minimizer, minimum = loss(M, minimizer), options = opt)
end

"""
    FrankWolfeOptimizer()

Manopt Frank-Wolfe optimizer for constrained manifold-valued problems.
"""
struct FrankWolfeOptimizer <: AbstractManoptOptimizer end

function call_manopt_optimizer(
        M::ManifoldsBase.AbstractManifold,
        opt::FrankWolfeOptimizer,
        loss,
        gradF,
        x0;
        hessF = nothing, # ignore that keyword for this solver
        kwargs...
    )
    opt = Frank_Wolfe_method(M, loss, gradF, x0; return_state = true, kwargs...)
    minimizer = Manopt.get_solver_result(opt)
    return (; minimizer = minimizer, minimum = loss(M, minimizer), options = opt)
end

## OptimizationBase.jl stuff
function SciMLBase.requiresgradient(
        opt::Union{
            GradientDescentOptimizer, ConjugateGradientDescentOptimizer,
            QuasiNewtonOptimizer, ConvexBundleOptimizer, FrankWolfeOptimizer,
            AdaptiveRegularizationCubicOptimizer, TrustRegionsOptimizer,
        }
    )
    return true
end
function SciMLBase.requireshessian(
        opt::Union{
            AdaptiveRegularizationCubicOptimizer, TrustRegionsOptimizer,
        }
    )
    return true
end

const GradientBasedManoptOptimizer = Union{
    GradientDescentOptimizer, ConjugateGradientDescentOptimizer,
    QuasiNewtonOptimizer, ConvexBundleOptimizer, FrankWolfeOptimizer,
    AdaptiveRegularizationCubicOptimizer, TrustRegionsOptimizer,
}

function _default_convergence_criterion(::GradientBasedManoptOptimizer, M, abstol)
    return Manopt.StopWhenGradientNormLess(abstol)
end

function _default_convergence_criterion(::AbstractManoptOptimizer, M, abstol)
    return Manopt.StopWhenChangeLess(M, abstol)
end

# Convergence criterion injected alongside `StopAfterIteration`/`StopAfter` only when the
# user supplies `maxiters`/`maxtime` without an explicit `abstol` or `stopping_criterion`
# (see `__map_optimizer_args!`). Each method mirrors, verbatim, the convergence part of the
# corresponding high-level Manopt solver's default `stopping_criterion` (checked against
# Manopt 0.5.25 and 0.6.6), so `maxiters`/`maxtime` behaves as if it merely retuned the
# iteration/time part of Manopt's default instead of discarding the whole thing.
#
# The remaining solvers return `nothing` and keep the pre-existing behavior (a
# `maxiters`-only run reports `ReturnCode.MaxIters`): the convergence-ish criteria in their
# Manopt defaults — `ParticleSwarm`'s `StopWhenSwarmVelocityLess`, `NelderMead`'s
# `StopWhenPopulationConcentrated`, `ConvexBundle`'s `StopWhenLagrangeMultiplierLess`, and
# parts of `CMAES`'s `default_cma_es_stopping_criterion` — all have
# `Manopt.indicates_convergence(...) == false`, so injecting them could never yield
# `ReturnCode.Success`; it would only cut the run short and turn `MaxIters` into `Failure`.
function _manopt_default_convergence_criterion(::GradientDescentOptimizer, M)
    return Manopt.StopWhenGradientNormLess(1.0e-8)
end
function _manopt_default_convergence_criterion(::ConjugateGradientDescentOptimizer, M)
    return Manopt.StopWhenGradientNormLess(1.0e-8)
end
function _manopt_default_convergence_criterion(::QuasiNewtonOptimizer, M)
    return Manopt.StopWhenGradientNormLess(1.0e-6)
end
function _manopt_default_convergence_criterion(::TrustRegionsOptimizer, M)
    return Manopt.StopWhenGradientNormLess(1.0e-6)
end
function _manopt_default_convergence_criterion(::AdaptiveRegularizationCubicOptimizer, M)
    # Manopt's default also contains `StopWhenAllLanczosVectorsUsed`, a safeguard rather
    # than a convergence test; it is not constructible independently of the sub-state, and
    # the always-present `StopAfterIteration` bounds the run in its stead.
    return Manopt.StopWhenGradientNormLess(1.0e-9)
end
function _manopt_default_convergence_criterion(::FrankWolfeOptimizer, M)
    # Frank-Wolfe solves constrained problems, where the gradient norm alone need not
    # vanish at a boundary optimum; Manopt's default therefore also stops on the change
    # between iterates, and dropping that half would leave such runs on `MaxIters`.
    return Manopt.StopWhenGradientNormLess(1.0e-8) | Manopt.StopWhenChangeLess(M, 1.0e-8)
end
_manopt_default_convergence_criterion(::AbstractManoptOptimizer, M) = nothing

function build_loss(f::OptimizationBase.OptimizationFunction, prob, cb)
    # TODO: I do not understand this. Why is the manifold not used?
    # Either this is an Euclidean cost, then we should probably still call `embed`,
    # or it is not, then we need M.
    return function (::AbstractManifold, θ)
        x = f.f(θ, prob.p)
        cb(x, θ)
        __x = first(x)
        return prob.sense === OptimizationBase.MaxSense ? -__x : __x
    end
end

function build_gradF(f::OptimizationBase.OptimizationFunction{true})
    function g(M::AbstractManifold, G, θ)
        f.grad(G, θ)
        return G .= riemannian_gradient(M, θ, G)
    end
    function g(M::AbstractManifold, θ)
        G = zero(θ)
        f.grad(G, θ)
        return riemannian_gradient(M, θ, G)
    end
    return g
end

function build_hessF(f::OptimizationBase.OptimizationFunction{true})
    function h(M::AbstractManifold, H1, θ, X)
        H = zeros(eltype(θ), length(θ))
        f.hv(H, θ, X)
        G = zeros(eltype(θ), length(θ))
        f.grad(G, θ)
        return riemannian_Hessian!(M, H1, θ, G, H, X)
    end
    function h(M::AbstractManifold, θ, X)
        H = zeros(eltype(θ), length(θ))
        f.hv(H, θ, X)
        G = zeros(eltype(θ), length(θ))
        f.grad(G, θ)
        return riemannian_Hessian(M, θ, G, H, X)
    end
    return h
end

function SciMLBase.__solve(cache::OptimizationBase.OptimizationCache{O}) where {O <: AbstractManoptOptimizer}
    local x, cur, state

    manifold = cache.manifold
    gradF = haskey(cache.solver_args, :riemannian_grad) ?
        cache.solver_args[:riemannian_grad] : nothing
    hessF = haskey(cache.solver_args, :riemannian_hess) ?
        cache.solver_args[:riemannian_hess] : nothing

    if manifold === nothing
        throw(ArgumentError("Manifold not specified in the problem for e.g. `OptimizationProblem(f, x, p; manifold = SymmetricPositiveDefinite(5))`."))
    end

    function _cb(x, θ)
        opt_state = OptimizationBase.OptimizationState(
            iter = 0,
            u = θ,
            p = cache.p,
            objective = x[1]
        )
        cb_call = cache.callback(opt_state, x...)
        if !(cb_call isa Bool)
            error("The callback should return a boolean `halt` for whether to stop the optimization process.")
        end
        return cb_call
    end
    solver_kwarg = __map_optimizer_args!(
        cache, cache.opt, manifold; callback = _cb,
        maxiters = cache.solver_args.maxiters,
        maxtime = cache.solver_args.maxtime,
        abstol = cache.solver_args.abstol,
        reltol = cache.solver_args.reltol,
        cache.solver_args...
    )

    _loss = build_loss(cache.f, cache, _cb)

    if gradF === nothing
        gradF = build_gradF(cache.f)
    end

    if hessF === nothing
        hessF = build_hessF(cache.f)
    end

    stopping_kwarg = if haskey(solver_kwarg, :stopping_criterion)
        (; stopping_criterion = Manopt.StopWhenAny(solver_kwarg.stopping_criterion...))
    else
        (;)
    end

    opt_res = call_manopt_optimizer(
        manifold, cache.opt, _loss, gradF, cache.u0;
        solver_kwarg..., stopping_kwarg..., hessF
    )

    asc = get_stopping_criterion(opt_res.options)
    active = Manopt.get_active_stopping_criteria(asc)
    opt_ret = if Manopt.has_converged(asc)
        ReturnCode.Success
    elseif any(c -> c isa Manopt.StopWhenChangeLess, active)
        # `indicates_convergence(StopWhenChangeLess)` is `false` in Manopt 0.5 but `true`
        # in Manopt 0.6 (where `has_converged` above already yields `Success`); map it
        # explicitly so the retcode does not depend on the Manopt version.
        ReturnCode.Success
    elseif any(c -> c isa Manopt.StopAfterIteration, active)
        ReturnCode.MaxIters
    elseif any(c -> c isa Manopt.StopAfter, active)
        ReturnCode.MaxTime
    elseif any(c -> c isa Union{Manopt.StopWhenCostNaN, Manopt.StopWhenIterateNaN}, active)
        ReturnCode.Unstable
    elseif any(c -> c isa Manopt.StopWhenStepsizeLess, active)
        ReturnCode.Stalled
    else
        ReturnCode.Failure
    end

    return SciMLBase.build_solution(
        cache,
        cache.opt,
        opt_res.minimizer,
        cache.sense === OptimizationBase.MaxSense ?
            -opt_res.minimum : opt_res.minimum;
        original = opt_res.options,
        retcode = opt_ret
    )
end

# The solvers this package owns. `AdaptiveRegularizationCubicOptimizer` and
# `TrustRegionsOptimizer` are wrappers just like the rest and are exported for the same
# reason; they were only reachable qualified before.
export AbstractManoptOptimizer,
    GradientDescentOptimizer, NelderMeadOptimizer, ConjugateGradientDescentOptimizer,
    ParticleSwarmOptimizer, QuasiNewtonOptimizer, CMAESOptimizer, ConvexBundleOptimizer,
    AdaptiveRegularizationCubicOptimizer, TrustRegionsOptimizer, FrankWolfeOptimizer

# Manopt's own surface is ~480 names and includes `solve!`, `BFGS`, `NelderMead` and
# `getindex`, so blanket-reexporting it was genuinely harmful and is not restored.
# Everything Optimization.jl's Manopt docs use from it — stepsizes, stopping criteria,
# quasi-Newton update rules — is spelled qualified (`Manopt.ArmijoLinesearch(...)`), so
# the module binding is what has to be in scope; exporting it removes the `using Manopt`
# the docs had to add.
export Manopt

end # module OptimizationManopt
