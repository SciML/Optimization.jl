module OptimizationODE

using Reexport
@reexport using Optimization, Optimization.SciMLBase
using DifferentialEquations
using Optimization.LinearAlgebra

export ODEOptimizer, ODEGradientDescent, RKChebyshevDescent, RKAccelerated, PRKChebyshevDescent

struct ODEOptimizer{T}
    solver::Type{T}
end

# Solver Constructors (users call these)
ODEGradientDescent()   = ODEOptimizer(Euler)
RKChebyshevDescent()   = ODEOptimizer(ROCK2)
RKAccelerated()        = ODEOptimizer(Tsit5)
PRKChebyshevDescent()  = ODEOptimizer(Vern7)


SciMLBase.requiresbounds(::ODEOptimizer)              = false
SciMLBase.allowsbounds(::ODEOptimizer)                = false
SciMLBase.allowscallback(::ODEOptimizer)              = true
SciMLBase.supports_opt_cache_interface(::ODEOptimizer) = true
SciMLBase.requiresgradient(::ODEOptimizer)            = true
SciMLBase.requireshessian(::ODEOptimizer)             = false
SciMLBase.requiresconsjac(::ODEOptimizer)             = false
SciMLBase.requiresconshess(::ODEOptimizer)            = false


function SciMLBase.__init(prob::OptimizationProblem, opt::ODEOptimizer, data=Optimization.DEFAULT_DATA;
    η=0.1, dt=nothing, tmax=100.0, callback=Optimization.DEFAULT_CALLBACK, progress=false,
    maxiters=nothing, kwargs...)

    return OptimizationCache(prob, opt, data;
        η=η, dt=dt, tmax=tmax, callback=callback, progress=progress,
        maxiters=maxiters, kwargs...)
end

function SciMLBase.__solve(
  cache::OptimizationCache{F,RC,LB,UB,LC,UC,S,O,D,P,C}
  ) where {F,RC,LB,UB,LC,UC,S,O<:ODEOptimizer,D,P,C}

    η     = get(cache.solver_args, :η, 0.1)
    dt    = get(cache.solver_args, :dt, nothing)
    tmax  = get(cache.solver_args, :tmax, 100.0)
    maxit = get(cache.solver_args, :maxiters, 1000)

    u0 = copy(cache.u0)
    p  = cache.p

    if cache.f.grad === nothing
        error("ODEOptimizer requires a gradient. Please provide a function with `grad` defined.")
    end

    function f!(du, u, p, t)
        cache.f.grad(du, u, p)
        @. du = -η * du
        return nothing
    end

    ss_prob = SteadyStateProblem(f!, u0, p)

    algorithm = DynamicSS(cache.opt.solver())

    cb = cache.callback
    if cb != Optimization.DEFAULT_CALLBACK || get(cache.solver_args,:progress,false) === true
        function condition(u, t, integrator)
            true
        end
        function affect!(integrator)
            u_now = integrator.u
            state = Optimization.OptimizationState(u=u_now, objective=cache.f(u_now, p))
            Optimization.callback_function(cb, state)
        end
        cb_struct = DiscreteCallback(condition, affect!)
        callback = CallbackSet(cb_struct)
    else
        callback = nothing
    end

    solve_kwargs = Dict{Symbol, Any}(:callback => callback)
    if !isnothing(maxit)
        solve_kwargs[:maxiters] = maxit
    end
    if dt !== nothing
        solve_kwargs[:dt] = dt
    end

    sol = solve(ss_prob, algorithm; solve_kwargs...)
has_destats = hasproperty(sol, :destats)
has_t = hasproperty(sol, :t) && !isempty(sol.t)

stats = Optimization.OptimizationStats(
    iterations = has_destats ? get(sol.destats, :iters, 10) : (has_t ? length(sol.t) - 1 : 10),
    time = has_t ? sol.t[end] : 0.0,
    fevals = has_destats ? get(sol.destats, :f_calls, 0) : 0,
    gevals = has_destats ? get(sol.destats, :iters, 0) : 0,
    hevals = 0
)

    SciMLBase.build_solution(cache, cache.opt, sol.u, cache.f(sol.u, p);
        retcode = ReturnCode.Success,
        stats = stats
    )
end

end
