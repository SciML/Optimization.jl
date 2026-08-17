module SimpleOptimization

using Reexport
# Not re-exported: the optimization API comes from `Optimization`/`OptimizationBase`,
# which the user loads directly. This package's public surface is its own solvers.
using OptimizationBase
using SciMLBase
using SciMLBase: _unwrap_val
using SimpleNonlinearSolve
using ADTypes
using LinearAlgebra
using CommonSolve
using LineSearch: StrongWolfeLineSearch

abstract type SimpleOptimizationAlgorithm end

"""
    SimpleLBFGS(; threshold::Union{Val, Int} = Val(10),
                linesearch = StrongWolfeLineSearch(; maxiters = 50, zoom_maxiters = 50))

A lightweight, loop-unrolled Limited-memory BFGS (L-BFGS) optimization algorithm.
This algorithm is designed for small-scale optimization problems where low overhead
is critical.

## Arguments

  - `threshold`: The number of past iterations to store for approximating the inverse
    Hessian. Default is `Val(10)`. Can be specified as either a `Val` type for compile-time
    optimization or an `Int`.
  - `linesearch`: A `LineSearch.StrongWolfeLineSearch` used for step-size selection.

## Description

`SimpleLBFGS` uses a limited-memory approximation to the BFGS update, storing only the
last `threshold` iterations of gradient information. This makes it memory-efficient
for problems with many variables while still achieving superlinear convergence.

Minimizes the objective with a Strong Wolfe line search. Supports box constraints
(`lb`/`ub`) via projection. `u0` must be statically sized (`SVector`, not `Vector`)
and the objective must be out-of-place (`OptimizationFunction{false}`).

`ReturnCode.Success` when the projected gradient meets the tolerance,
`ReturnCode.MaxIters` when the iteration limit is hit, and `ReturnCode.Failure`
on a non-finite iterate or a failed line search.

## Example

```julia
using SimpleOptimization, Optimization, ForwardDiff, StaticArrays

rosenbrock(x, p) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
x0 = SVector(0.0, 0.0)
optf = OptimizationFunction{false}(rosenbrock, Optimization.AutoForwardDiff())
prob = OptimizationProblem{false}(optf, x0)
sol = solve(prob, SimpleLBFGS())
```
"""
struct SimpleLBFGS{Threshold, LS} <: SimpleOptimizationAlgorithm
    linesearch::LS
end

__get_threshold(::SimpleLBFGS{threshold}) where {threshold} = Val(threshold)
function SimpleLBFGS(;
        threshold::Union{Val, Int} = Val(10),
        linesearch = StrongWolfeLineSearch(; maxiters = 50, zoom_maxiters = 50)
    )
    return SimpleLBFGS{_unwrap_val(threshold), typeof(linesearch)}(linesearch)
end
SciMLBase.allowsbounds(::SimpleLBFGS) = true

"""
    SimpleBFGS()

A lightweight, loop-unrolled BFGS optimization algorithm. This algorithm is designed
for small-scale unconstrained optimization problems where low overhead is critical.

## Description

`SimpleBFGS` implements the Broyden-Fletcher-Goldfarb-Shanno (BFGS) quasi-Newton method.
It builds an approximation to the inverse Hessian matrix using gradient information,
achieving superlinear convergence for smooth objective functions.

Internally, it wraps `SimpleBroyden` from SimpleNonlinearSolve.jl to find the root of
the gradient (i.e., the stationary point of the objective).

## Example

```julia
using SimpleOptimization, Optimization, ForwardDiff

rosenbrock(x, p) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
x0 = zeros(2)
optf = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, x0)
sol = solve(prob, SimpleBFGS())
```
"""
struct SimpleBFGS <: SimpleOptimizationAlgorithm end

"""
    SimpleGradientDescent(; eta = 0.01)

A lightweight gradient descent optimization algorithm. This algorithm is designed
for small-scale unconstrained optimization problems where low overhead is critical.

## Arguments

  - `eta`: The learning rate (step size). Default is `0.01`.

## Description

`SimpleGradientDescent` implements the steepest descent method, updating the iterate
via `x_{k+1} = x_k - eta * gradient(f, x_k)` at each step. While it has only linear
convergence, it is the simplest first-order method and is useful as a baseline or for
problems where quasi-Newton overhead is undesirable.

## Example

```julia
using SimpleOptimization, Optimization, ForwardDiff

rosenbrock(x, p) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
x0 = zeros(2)
optf = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, x0)
sol = solve(prob, SimpleGradientDescent(; eta = 0.001), maxiters = 10000)
```
"""
struct SimpleGradientDescent{T} <: SimpleOptimizationAlgorithm
    eta::T
end

SimpleGradientDescent(; eta = 0.01) = SimpleGradientDescent(eta)

"""
    SimpleNewton()

A lightweight Newton optimization algorithm. This algorithm is designed for small-scale
unconstrained optimization problems where quadratic convergence is desired.

## Description

`SimpleNewton` implements Newton's method for optimization, which finds a stationary point
by solving the system `gradient(f, x) = 0` using Newton-Raphson iteration. This requires
computing the Hessian (via automatic differentiation of the gradient) and gives quadratic
convergence near the solution for smooth objective functions.

Internally, it wraps `SimpleNewtonRaphson` from SimpleNonlinearSolve.jl to find the root
of the gradient. The Hessian is computed automatically by SimpleNewtonRaphson's internal
AD applied to the gradient function.

## Example

```julia
using SimpleOptimization, Optimization, ForwardDiff

rosenbrock(x, p) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
x0 = [0.5, 0.5]
optf = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, x0)
sol = solve(prob, SimpleNewton())
```
"""
struct SimpleNewton <: SimpleOptimizationAlgorithm end

SciMLBase.has_init(::SimpleOptimizationAlgorithm) = true

export SimpleBFGS, SimpleLBFGS, SimpleGradientDescent, SimpleNewton, SimpleSOAP

# Source: https://github.com/SciML/Optimization.jl/blob/9c5070b3db838e05794ded348b8b17df0f9e38c1/src/function.jl#L104
function instantiate_gradient(f, adtype::ADTypes.AbstractADType)
    adtypestr = string(adtype)
    _strtind = findfirst('.', adtypestr)
    strtind = isnothing(_strtind) ? 5 : _strtind + 5
    open_nrmlbrkt_ind = findfirst('(', adtypestr)
    open_squigllybrkt_ind = findfirst('{', adtypestr)
    open_brkt_ind = if isnothing(open_squigllybrkt_ind)
        open_nrmlbrkt_ind
    elseif isnothing(open_nrmlbrkt_ind)
        open_squigllybrkt_ind
    else
        min(open_nrmlbrkt_ind, open_squigllybrkt_ind)
    end
    lastidx = isnothing(open_brkt_ind) ? lastindex(adtypestr) : (open_brkt_ind - 1)
    adpkg = adtypestr[strtind:lastidx]
    throw(ArgumentError("The passed automatic differentiation backend choice is not available. Please load the corresponding AD package $adpkg."))
end

@inline _as_u(x, g) = convert(typeof(x), g)

@inline _project(x, ::Nothing, ::Nothing) = x
@inline _project(x, lb, ub) = clamp.(x, lb, ub)

@inline _projected_gradient(x, g, ::Nothing, ::Nothing) = g
@inline _projected_gradient(x, g, lb, ub) = x - _project(x - g, lb, ub)

@inline _feasible_direction(x, direction, ::Nothing, ::Nothing) = direction
@inline function _feasible_direction(x, direction, lb, ub)
    return map(x, direction, lb, ub) do xi, di, li, ui
        ((xi ≤ li && di < 0) || (xi ≥ ui && di > 0)) ? zero(di) : di
    end
end

@inline _max_feasible_step(x, direction, ::Nothing, ::Nothing, α_max) = α_max
@inline function _max_feasible_step(x, direction, lb, ub, α_max)
    α = α_max
    for i in eachindex(x)
        if direction[i] > 0
            α = min(α, (ub[i] - x[i]) / direction[i])
        elseif direction[i] < 0
            α = min(α, (lb[i] - x[i]) / direction[i])
        end
    end
    return max(α, zero(α))
end

@inline function _objective_line_search(
        ls_cache, f, grad_f, p, x, fx, g, direction, lb, ub
    )
    T = eltype(x)
    α_max = _max_feasible_step(x, direction, lb, ub, T(ls_cache.α_max))
    α_max ≤ zero(T) && return x, fx, g, false

    ls = CommonSolve.solve!(ls_cache, x, direction; α_max)
    ls.retcode == SciMLBase.ReturnCode.Success || return x, fx, g, false

    candidate = _project(x + T(ls.step_size) * direction, lb, ub)
    return candidate, T(f(candidate, p)), _as_u(x, grad_f(candidate, p)), true
end

@inline function _lbfgs_direction(g, s_hist, y_hist, pseudo_iteration, ::Val{M}) where {M}
    T = eltype(g)
    α = ntuple(_ -> zero(T), Val(M))
    q = g
    lower = pseudo_iteration - M
    upper = pseudo_iteration - 1

    for index in upper:-1:lower
        index < 1 && continue
        j = mod1(index, M)
        s, y = s_hist[j], y_hist[j]
        ρ = inv(dot(y, s))
        α = Base.setindex(α, ρ * dot(s, q), j)
        q -= α[j] * y
    end

    r = q
    if pseudo_iteration > 1
        j = mod1(upper, M)
        s, y = s_hist[j], y_hist[j]
        r = (dot(s, y) / sum(abs2, y)) * q
    end

    for index in lower:upper
        index < 1 && continue
        j = mod1(index, M)
        s, y = s_hist[j], y_hist[j]
        ρ = inv(dot(y, s))
        r += s * (α[j] - ρ * dot(y, r))
    end
    return -r
end

@inline function _store_history(s_hist, y_hist, pseudo_iteration, s, y, ::Val{M}) where {M}
    j = mod1(pseudo_iteration, M)
    return Base.setindex(s_hist, s, j), Base.setindex(y_hist, y, j)
end

@inline function _lbfgs(
        grad_f, f, p, x0, lb, ub,
        maxiters, abstol, reltol, linesearch, ::Val{M}
    ) where {M}
    T = eltype(x0)
    x = _project(x0, lb, ub)
    fx = T(f(x, p))
    g = _as_u(x, grad_f(x, p))
    default_tol = sqrt(eps(T))
    abs_tol = abstol === nothing ? default_tol : max(T(abstol), default_tol)
    rel_tol = reltol === nothing ? zero(T) : max(T(reltol), zero(T))
    initial_residual = maximum(abs, _projected_gradient(x, g, lb, ub))
    tol = max(abs_tol, rel_tol * initial_residual)
    zero_x = zero(x)
    s_hist = ntuple(_ -> zero_x, Val(M))
    y_hist = ntuple(_ -> zero_x, Val(M))
    pseudo_iteration = 0
    retcode = SciMLBase.ReturnCode.MaxIters
    iters = 0

    optf = OptimizationFunction{false}(f; grad = grad_f)
    ls_cache = CommonSolve.init(
        OptimizationProblem{false}(optf, x, p), linesearch, fx, x
    )

    for i in 1:maxiters
        if !isfinite(fx) || !all(isfinite, g)
            retcode = SciMLBase.ReturnCode.Failure
            break
        end
        if maximum(abs, _projected_gradient(x, g, lb, ub)) ≤ tol
            retcode = SciMLBase.ReturnCode.Success
            break
        end
        iters = i

        pseudo_iteration += 1
        direction = _feasible_direction(
            x, _lbfgs_direction(g, s_hist, y_hist, pseudo_iteration, Val(M)), lb, ub
        )
        if dot(direction, g) ≥ zero(T)
            pseudo_iteration = 1
            direction = _feasible_direction(x, -g, lb, ub)
        end
        if maximum(abs, direction) ≤ eps(T)
            retcode = SciMLBase.ReturnCode.Failure
            break
        end

        x_next, fx_next, g_next, accepted = _objective_line_search(
            ls_cache, f, grad_f, p, x, fx, g, direction, lb, ub
        )
        if !accepted
            retcode = SciMLBase.ReturnCode.Failure
            break
        end

        s, y = x_next - x, g_next - g
        curvature = dot(s, y)
        curvature_floor = sqrt(eps(T)) * max(norm(s) * norm(y), eps(T))
        if !isfinite(curvature) || curvature ≤ curvature_floor
            pseudo_iteration = 0
        else
            s_hist, y_hist = _store_history(
                s_hist, y_hist, pseudo_iteration, s, y, Val(M)
            )
        end
        x, fx, g = x_next, fx_next, g_next
    end
    if retcode == SciMLBase.ReturnCode.MaxIters
        if !isfinite(fx) || !all(isfinite, g)
            retcode = SciMLBase.ReturnCode.Failure
        elseif maximum(abs, _projected_gradient(x, g, lb, ub)) ≤ tol
            retcode = SciMLBase.ReturnCode.Success
        end
    end
    return x, fx, retcode, iters
end

function SciMLBase.__solve(cache::OptimizationCache{O}) where {O <: SimpleLBFGS}
    maxiters = OptimizationBase._check_and_convert_maxiters(cache.solver_args.maxiters)
    if maxiters === nothing
        maxiters = 100
    end

    abstol = cache.solver_args.abstol
    reltol = cache.solver_args.reltol

    u0 = cache.u0
    u0 isa Array && throw(
        ArgumentError(
            "SimpleLBFGS requires a statically sized `u0` (e.g. `SVector`). Got $(typeof(u0))."
        )
    )
    U = typeof(u0)
    T = eltype(u0)

    ∇f_inner = instantiate_gradient(Base.Fix2(cache.f.f, cache.p), cache.f.adtype)
    grad_f = (u, _) -> convert(U, ∇f_inner(u))

    ls = cache.opt.linesearch
    ls isa StrongWolfeLineSearch ||
        throw(ArgumentError("SimpleLBFGS requires a StrongWolfeLineSearch"))
    typed_linesearch = StrongWolfeLineSearch(;
        autodiff = ls.autodiff,
        c1 = T(ls.c1), c2 = T(ls.c2),
        α_init = T(ls.α_init), α_max = T(ls.α_max),
        maxiters = ls.maxiters, zoom_maxiters = ls.zoom_maxiters
    )

    lb = cache.lb === nothing ? nothing : convert(U, cache.lb)
    ub = cache.ub === nothing ? nothing : convert(U, cache.ub)

    t0 = time()
    u, objective, retcode, iters = _lbfgs(
        grad_f, cache.f.f, cache.p, u0, lb, ub, maxiters, abstol, reltol,
        typed_linesearch, __get_threshold(cache.opt)
    )
    return SciMLBase.build_solution(
        cache, cache.opt, u, objective;
        retcode = retcode,
        stats = OptimizationBase.OptimizationStats(; iterations = iters, time = time() - t0)
    )
end

function SciMLBase.__solve(cache::OptimizationCache{O}) where {O <: SimpleBFGS}
    maxiters = OptimizationBase._check_and_convert_maxiters(cache.solver_args.maxiters)
    if maxiters === nothing
        maxiters = 100
    end

    abstol = cache.solver_args.abstol
    reltol = cache.solver_args.reltol

    f = Base.Fix2(cache.f.f, cache.p)
    adtype = cache.f.adtype
    ∇f_inner = instantiate_gradient(f, adtype)
    # Wrap gradient to take (u, p) as NonlinearProblem expects
    ∇f = (u, _) -> ∇f_inner(u)

    nlprob = NonlinearProblem(∇f, cache.u0)
    nlsol = solve(
        nlprob,
        SimpleBroyden(; linesearch = nothing);
        maxiters = maxiters,
        abstol = abstol,
        reltol = reltol
    )
    θ = nlsol.u

    stats = OptimizationBase.OptimizationStats(;
        iterations = maxiters,
        time = 0.0,
        fevals = 0
    )
    return SciMLBase.build_solution(
        cache, cache.opt,
        θ,
        cache.f(θ, cache.p);
        original = nlsol,
        retcode = nlsol.retcode,
        stats = stats
    )
end

function SciMLBase.__solve(cache::OptimizationCache{O}) where {O <: SimpleGradientDescent}
    maxiters = OptimizationBase._check_and_convert_maxiters(cache.solver_args.maxiters)
    if maxiters === nothing
        maxiters = 1000
    end

    abstol = cache.solver_args.abstol
    if abstol === nothing
        abstol = 1.0e-8
    end

    f = Base.Fix2(cache.f.f, cache.p)
    adtype = cache.f.adtype
    ∇f = instantiate_gradient(f, adtype)
    η = cache.opt.eta

    θ = copy(cache.u0)
    g = ∇f(θ)
    retcode = SciMLBase.ReturnCode.MaxIters
    iters = maxiters
    for i in 1:maxiters
        θ = θ .- η .* g
        g = ∇f(θ)
        if norm(g) < abstol
            retcode = SciMLBase.ReturnCode.Success
            iters = i
            break
        end
    end

    stats = OptimizationBase.OptimizationStats(;
        iterations = iters,
        time = 0.0,
        fevals = iters + 1
    )
    return SciMLBase.build_solution(
        cache, cache.opt,
        θ,
        cache.f(θ, cache.p);
        retcode = retcode,
        stats = stats
    )
end

function SciMLBase.__solve(cache::OptimizationCache{O}) where {O <: SimpleNewton}
    maxiters = OptimizationBase._check_and_convert_maxiters(cache.solver_args.maxiters)
    if maxiters === nothing
        maxiters = 100
    end

    abstol = cache.solver_args.abstol
    reltol = cache.solver_args.reltol

    f = Base.Fix2(cache.f.f, cache.p)
    adtype = cache.f.adtype
    ∇f_inner = instantiate_gradient(f, adtype)
    # Wrap gradient to take (u, p) as NonlinearProblem expects
    ∇f = (u, _) -> ∇f_inner(u)

    nlprob = NonlinearProblem(∇f, cache.u0)
    nlsol = solve(
        nlprob,
        SimpleNewtonRaphson();
        maxiters = maxiters,
        abstol = abstol,
        reltol = reltol
    )
    θ = nlsol.u

    stats = OptimizationBase.OptimizationStats(;
        iterations = maxiters,
        time = 0.0,
        fevals = 0
    )
    return SciMLBase.build_solution(
        cache, cache.opt,
        θ,
        cache.f(θ, cache.p);
        original = nlsol,
        retcode = nlsol.retcode,
        stats = stats
    )
end

"""
    SimpleSOAP(; eta=3e-3, beta=(0.95, 0.95), shampoo_beta=-1.0, epsilon=1e-8,
                 freq=10, max_dim=10000, weight_decay=0.01)

SOAP optimizer (ShampoO with Adam in the Preconditioner's eigenbasis).
For matrix-valued parameters, runs AdamW in the eigenbasis of Shampoo's
preconditioner. For vector-valued parameters, falls back to standard AdamW.

Based on "SOAP: Improving and Stabilizing Shampoo using Adam"
(https://arxiv.org/abs/2409.11321).

## Arguments

  - `eta`: learning rate (default: 3e-3)
  - `beta`: (β₁, β₂) for Adam momentum and second moment (default: (0.95, 0.95))
  - `shampoo_beta`: separate β for preconditioner EMA; if < 0, uses β₂ (default: -1)
  - `epsilon`: numerical stability constant (default: 1e-8)
  - `freq`: how often to recompute eigenbasis (default: 10)
  - `max_dim`: dimensions larger than this use identity rotation (default: 10000)
  - `weight_decay`: decoupled weight decay, applied as `lr * wd` (default: 0.01)

## Example

```julia
using SimpleOptimization, ForwardDiff

f(x, p) = sum(abs2, x .- p)
W0 = randn(8, 8)
optf = OptimizationFunction(f, AutoForwardDiff())
prob = OptimizationProblem(optf, W0, ones(8, 8))
sol = solve(prob, SimpleSOAP(), maxiters = 500)
```
"""
struct SimpleSOAP{T} <: SimpleOptimizationAlgorithm
    eta::T
    beta::Tuple{T, T}
    shampoo_beta::T
    epsilon::T
    freq::Int
    max_dim::Int
    weight_decay::T
end

function SimpleSOAP(;
        eta = 3.0e-3, beta = (0.95, 0.95), shampoo_beta = -1.0,
        epsilon = 1.0e-8, freq = 10, max_dim = 10000, weight_decay = 0.01
    )
    T = promote_type(typeof(eta), typeof(epsilon), typeof(weight_decay))
    return SimpleSOAP(
        T(eta), T.(beta), T(shampoo_beta), T(epsilon), freq, max_dim, T(weight_decay)
    )
end

function SciMLBase.__solve(cache::OptimizationCache{O}) where {O <: SimpleSOAP}
    maxiters = OptimizationBase._check_and_convert_maxiters(cache.solver_args.maxiters)
    if maxiters === nothing
        maxiters = 1000
    end

    f = Base.Fix2(cache.f.f, cache.p)
    adtype = cache.f.adtype
    ∇f = instantiate_gradient(f, adtype)

    opt = cache.opt
    θ = copy(cache.u0)

    if ndims(θ) == 2
        _soap_solve_matrix!(θ, ∇f, opt, maxiters)
        geval = maxiters + 1
    else
        _soap_solve_vector!(θ, ∇f, opt, maxiters)
        geval = maxiters
    end

    stats = OptimizationBase.OptimizationStats(;
        iterations = maxiters,
        time = 0.0,
        fevals = geval
    )
    return SciMLBase.build_solution(
        cache, cache.opt,
        θ,
        cache.f(θ, cache.p);
        retcode = SciMLBase.ReturnCode.MaxIters,
        stats = stats
    )
end

function _soap_solve_vector!(θ, ∇f, opt, maxiters)
    T = eltype(θ)
    β1, β2 = T.(opt.beta)
    ε = T(opt.epsilon)
    η = T(opt.eta)
    wd = T(opt.weight_decay)

    ea = zero(θ)
    eas = zero(θ)
    g = similar(θ)

    for t in 1:maxiters
        g .= ∇f(θ)
        @. ea = β1 * ea + (1 - β1) * g
        @. eas = β2 * eas + (1 - β2) * g^2
        s = η * sqrt(1 - β2^t) / (1 - β1^t)
        @. θ = θ - s * ea / (sqrt(eas) + ε) - η * wd * θ
    end
    return nothing
end

function _soap_solve_matrix!(θ, ∇f, opt, maxiters)
    T = eltype(θ)
    m, n = size(θ)
    β1, β2 = T.(opt.beta)
    sβ = opt.shampoo_beta >= 0 ? T(opt.shampoo_beta) : β2
    ε = T(opt.epsilon)
    η = T(opt.eta)
    wd = T(opt.weight_decay)

    ea = zero(θ)                                       # first moment (rotated)
    eas = zero(θ)                                      # second moment (rotated)
    g = similar(θ)                                     # gradient
    buf = similar(θ)                                   # projection scratch
    tmp = similar(θ)                                   # mul! intermediate
    ea_buf = similar(θ)                                # adam step / permutation swap
    ea_orig = similar(θ)                               # momentum in original space

    uL = m <= opt.max_dim
    uR = n <= opt.max_dim

    L = uL ? fill!(similar(θ, T, m, m), zero(T)) : nothing  # left preconditioner G*G'
    R = uR ? fill!(similar(θ, T, n, n), zero(T)) : nothing  # right preconditioner G'*G
    QL = uL ? _soap_eye(θ, T, m) : nothing                  # left eigenbasis
    QR = uR ? _soap_eye(θ, T, n) : nothing                  # right eigenbasis
    s1L = uL ? similar(θ, T, m, m) : nothing                # left QR scratch
    s2L = uL ? similar(θ, T, m, m) : nothing                # left QR scratch
    eyeL = uL ? _soap_eye(θ, T, m) : nothing               # left identity (QR materialization)
    s1R = uR ? similar(θ, T, n, n) : nothing                # right QR scratch
    s2R = uR ? similar(θ, T, n, n) : nothing                # right QR scratch
    eyeR = uR ? _soap_eye(θ, T, n) : nothing               # right identity (QR materialization)

    q_ready = false

    for t in 0:maxiters
        g .= ∇f(θ)

        # First call: seed preconditioners, compute eigenbasis, skip update.
        if !q_ready
            _soap_accum!(L, R, g, sβ, uL, uR)
            uL && (QL .= _soap_eigh(L))
            uR && (QR .= _soap_eigh(R))
            q_ready = true
            continue
        end

        # Project gradient into eigenbasis
        _soap_fwd!(buf, g, QL, QR, uL, uR, tmp)

        # Update moments in rotated space
        @. ea = β1 * ea + (1 - β1) * buf
        @. eas = β2 * eas + (1 - β2) * buf^2

        # Adam update in rotated space, project back, apply
        @. ea_buf = ea / (sqrt(eas) + ε)
        _soap_bwd!(buf, ea_buf, QL, QR, uL, uR, tmp)
        s = η * sqrt(1 - β2^t) / (1 - β1^t)
        @. θ = θ - s * buf - η * wd * θ

        # Un-project momentum, accumulate preconditioners
        _soap_bwd!(ea_orig, ea, QL, QR, uL, uR, tmp)
        _soap_accum!(L, R, g, sβ, uL, uR)

        # Periodically update eigenbasis via power iteration + QR
        if t > 0 && t % opt.freq == 0
            uL && _soap_pqr!(QL, L, s1L, s2L, eyeL, eas, ea_buf, Val(:row))
            uR && _soap_pqr!(QR, R, s1R, s2R, eyeR, eas, ea_buf, Val(:col))
        end

        # Re-project momentum into (possibly updated) eigenbasis
        _soap_fwd!(ea, ea_orig, QL, QR, uL, uR, tmp)
    end
    return nothing
end

function _soap_eye(ref, ::Type{T}, n) where {T}
    A = fill!(similar(ref, T, n, n), zero(T))
    A[diagind(A)] .= one(T)
    return A
end

function _soap_accum!(L, R, G, sβ, uL, uR)
    a = 1 - sβ
    b = sβ
    uL && mul!(L, G, G', a, b)                 # L = sβ*L + (1-sβ)*G*G'
    uR && mul!(R, G', G, a, b)                 # R = sβ*R + (1-sβ)*G'*G
    return nothing
end

function _soap_eigh(P)
    T = eltype(P)
    S = Symmetric((P .+ P') ./ 2 + T(1.0e-30) * I)
    E = eigen(S)
    return E.vectors[:, end:-1:1]              # descending eigenvalue order
end

function _soap_pqr!(Q, GG, s1, s2, eye, eas, eas_buf, ::Val{dim}) where {dim}
    mul!(s1, GG, Q)                             # power iteration: GG * Q
    mul!(s2, Q', s1)                            # eigenvalue estimates: Q' * GG * Q
    perm = sortperm(diag(s2); rev = true)       # sort by descending eigenvalue
    s2 .= @view Q[:, perm]                      # reorder eigenvectors
    mul!(s1, GG, s2)                            # power iteration on sorted Q
    F = qr!(s1)                                 # orthogonalize
    mul!(Q, F.Q, eye)                           # materialize Q factor
    eas_buf .= eas                              # permute v to match new eigenvector order
    if dim === :row
        eas .= @view eas_buf[perm, :]
    else
        eas .= @view eas_buf[:, perm]
    end
    return nothing
end

function _soap_fwd!(out, X, QL, QR, uL, uR, tmp)   # out = QL' * X * QR
    if uL && uR
        mul!(tmp, QL', X)
        mul!(out, tmp, QR)
    elseif uL
        mul!(out, QL', X)
    elseif uR
        mul!(out, X, QR)
    else
        out .= X
    end
    return nothing
end

function _soap_bwd!(out, X, QL, QR, uL, uR, tmp)   # out = QL * X * QR'
    if uL && uR
        mul!(tmp, QL, X)
        mul!(out, tmp, QR')
    elseif uL
        mul!(out, QL, X)
    elseif uR
        mul!(out, X, QR')
    else
        out .= X
    end
    return nothing
end

end
