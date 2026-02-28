module SimpleOptimization

using Reexport
@reexport using OptimizationBase
using SciMLBase
using SciMLBase: _unwrap_val
using SimpleNonlinearSolve
using ADTypes
using LinearAlgebra: norm

abstract type SimpleOptimizationAlgorithm end

"""
    SimpleLBFGS(; threshold::Union{Val, Int} = Val(10))

A lightweight, loop-unrolled Limited-memory BFGS (L-BFGS) optimization algorithm.
This algorithm is designed for small-scale unconstrained optimization problems where
low overhead is critical.

## Arguments

  - `threshold`: The number of past iterations to store for approximating the inverse
    Hessian. Default is `Val(10)`. Can be specified as either a `Val` type for compile-time
    optimization or an `Int`.

## Description

`SimpleLBFGS` uses a limited-memory approximation to the BFGS update, storing only the
last `threshold` iterations of gradient information. This makes it memory-efficient
for problems with many variables while still achieving superlinear convergence.

Internally, it wraps `SimpleLimitedMemoryBroyden` from SimpleNonlinearSolve.jl to find
the root of the gradient (i.e., the stationary point of the objective).

## Example

```julia
using SimpleOptimization, Optimization, ForwardDiff

rosenbrock(x, p) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
x0 = zeros(2)
optf = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, x0)
sol = solve(prob, SimpleLBFGS())
```
"""
struct SimpleLBFGS{Threshold} <: SimpleOptimizationAlgorithm end

__get_threshold(::SimpleLBFGS{threshold}) where {threshold} = Val(threshold)
SimpleLBFGS(; threshold::Union{Val, Int} = Val(10)) = SimpleLBFGS{_unwrap_val(threshold)}()

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

export SimpleBFGS, SimpleLBFGS, SimpleGradientDescent, SimpleNewton

# Source: https://github.com/SciML/Optimization.jl/blob/9c5070b3db838e05794ded348b8b17df0f9e38c1/src/function.jl#L104
function instantiate_gradient(f, adtype::ADTypes.AbstractADType)
    adtypestr = string(adtype)
    _strtind = findfirst('.', adtypestr)
    strtind = isnothing(_strtind) ? 5 : _strtind + 5
    open_nrmlbrkt_ind = findfirst('(', adtypestr)
    open_squigllybrkt_ind = findfirst('{', adtypestr)
    open_brkt_ind = isnothing(open_squigllybrkt_ind) ? open_nrmlbrkt_ind :
        min(open_nrmlbrkt_ind, open_squigllybrkt_ind)
    adpkg = adtypestr[strtind:(open_brkt_ind - 1)]
    throw(ArgumentError("The passed automatic differentiation backend choice is not available. Please load the corresponding AD package $adpkg."))
end

function SciMLBase.__solve(cache::OptimizationCache{O}) where {O <: SimpleLBFGS}
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
        SimpleLimitedMemoryBroyden(;
            threshold = __get_threshold(cache.opt),
            linesearch = Val(false)
        );
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
        SimpleBroyden(; linesearch = Val(false));
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
        abstol = 1e-8
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

end
