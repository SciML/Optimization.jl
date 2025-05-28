module OptimizationODE

using Reexport
@reexport using Optimization, Optimization.SciMLBase
using DifferentialEquations

export ODEOptimizer, ODEGradientDescent, RKChebyshevDescent, RKAccelerated, PRKChebyshevDescent

abstract type AbstractODEOptimizer end

struct ODEOptimizer{T} <: AbstractODEOptimizer
    alg::T
end


const ODEGradientDescent  = ODEOptimizer(Euler())
const RKChebyshevDescent  = ODEOptimizer(ROCK2())
const RKAccelerated       = ODEOptimizer(Tsit5())
const PRKChebyshevDescent = ODEOptimizer(ROCK4())


SciMLBase.supports_opt_cache_interface(::ODEOptimizer) = true
SciMLBase.requiresgradient(::ODEOptimizer)             = true

function Optimization.__map_optimizer_args(cache::OptimizationCache, opt::ODEOptimizer;
        dt::Real = 0.01,
        maxiters::Integer = 100,
        callback = nothing,
        progress = false,
        kwargs...
    )
    cache.meta[:dt]      = dt
    cache.meta[:maxiters]= maxiters
    cache.meta[:callback]= callback
    cache.meta[:progress]= progress
    return nothing
end

function SciMLBase.__solve(
    cache::OptimizationCache{F,RC,LB,UB,LC,UC,S,<:ODEOptimizer,D,P,C}
) where {F,RC,LB,UB,LC,UC,S,D,P,C}
    dt = cache.solver_args[:dt]
    maxiters = cache.solver_args[:maxiters]
    tspan = (0.0, maxiters * dt)

    alg = cache.opt.alg

    prob = SteadyStateProblem(
        (du, u, p, t) -> begin
            cache.f.grad(du, u, cache.p)
            du .*= -1
        end,
        cache.u0,
        cache.p
    )

    sol = solve(prob, DynamicSS(alg); dt=dt)

    return SciMLBase.build_solution(cache, cache.opt, sol.u,
        sol.resid; original = sol, retcode = sol.retcode)
end

end
