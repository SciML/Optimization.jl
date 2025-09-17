module OptimizationODE

using Reexport
@reexport using Optimization, SciMLBase
using LinearAlgebra, ForwardDiff

using NonlinearSolve
using OrdinaryDiffEq, SteadyStateDiffEq

export ODEOptimizer, ODEGradientDescent, RKChebyshevDescent, RKAccelerated, HighOrderDescent
export DAEOptimizer, DAEMassMatrix

struct ODEOptimizer{T}
    solver::T
end

ODEGradientDescent() = ODEOptimizer(Euler())
RKChebyshevDescent() = ODEOptimizer(ROCK2())
RKAccelerated() = ODEOptimizer(Tsit5())
HighOrderDescent() = ODEOptimizer(Vern7())

struct DAEOptimizer{T}
    solver::T
end

DAEMassMatrix() = DAEOptimizer(Rodas5P(autodiff = false))


SciMLBase.requiresbounds(::ODEOptimizer) = false
SciMLBase.allowsbounds(::ODEOptimizer) = false
SciMLBase.allowscallback(::ODEOptimizer) = true
SciMLBase.supports_opt_cache_interface(::ODEOptimizer) = true
SciMLBase.requiresgradient(::ODEOptimizer) = true
SciMLBase.requireshessian(::ODEOptimizer) = false
SciMLBase.requiresconsjac(::ODEOptimizer) = false
SciMLBase.requiresconshess(::ODEOptimizer) = false


SciMLBase.requiresbounds(::DAEOptimizer) = false
SciMLBase.allowsbounds(::DAEOptimizer) = false
SciMLBase.allowsconstraints(::DAEOptimizer) = true
SciMLBase.allowscallback(::DAEOptimizer) = true
SciMLBase.supports_opt_cache_interface(::DAEOptimizer) = true
SciMLBase.requiresgradient(::DAEOptimizer) = true
SciMLBase.requireshessian(::DAEOptimizer) = false
SciMLBase.requiresconsjac(::DAEOptimizer) = true
SciMLBase.requiresconshess(::DAEOptimizer) = false


function SciMLBase.__init(prob::OptimizationProblem, opt::ODEOptimizer;
    callback=Optimization.DEFAULT_CALLBACK, progress=false, dt=nothing,
    maxiters=nothing, kwargs...)
    return OptimizationCache(prob, opt; callback=callback, progress=progress, dt=dt,
        maxiters=maxiters, kwargs...)
end

function SciMLBase.__init(prob::OptimizationProblem, opt::DAEOptimizer;
    callback=Optimization.DEFAULT_CALLBACK, progress=false, dt=nothing,
    maxiters=nothing, kwargs...)
    return OptimizationCache(prob, opt; callback=callback, progress=progress, dt=dt,
        maxiters=maxiters, kwargs...)
end

function SciMLBase.__solve(
    cache::OptimizationCache{F,RC,LB,UB,LC,UC,S,O,D,P,C}
    ) where {F,RC,LB,UB,LC,UC,S,O<:Union{ODEOptimizer,DAEOptimizer},D,P,C}

    dt = get(cache.solver_args, :dt, nothing)
    maxit = get(cache.solver_args, :maxiters, nothing)
    u0 = copy(cache.u0)
    p = cache.p # Properly handle NullParameters

    if cache.opt isa ODEOptimizer
        return solve_ode(cache, dt, maxit, u0, p)
    else
        if cache.opt.solver isa SciMLBase.AbstractDAEAlgorithm
            return solve_dae_implicit(cache, dt, maxit, u0, p)
        else
            return solve_dae_mass_matrix(cache, dt, maxit, u0, p)
        end
    end
end

function solve_ode(cache, dt, maxit, u0, p)
    if cache.f.grad === nothing
        error("ODEOptimizer requires a gradient. Please provide a function with `grad` defined.")
    end

    function f!(du, u, p, t)
        cache.f.grad(du, u, p)
        @. du = -du
        return nothing
    end

    ss_prob = SteadyStateProblem(f!, u0, p)

    algorithm = DynamicSS(cache.opt.solver)

    if cache.callback !== Optimization.DEFAULT_CALLBACK
        condition = (u, t, integrator) -> true
        affect! = (integrator) -> begin
            u_opt = integrator.u isa AbstractArray ? integrator.u : integrator.u.u
            l = cache.f(integrator.u, integrator.p)
            cache.callback(integrator.u, l)
        end
        cb = DiscreteCallback(condition, affect!)
        solve_kwargs = Dict{Symbol, Any}(:callback => cb)
    else
        solve_kwargs = Dict{Symbol, Any}()
    end
    
    if !isnothing(maxit)
        solve_kwargs[:maxiters] = maxit
    end
    if dt !== nothing
        solve_kwargs[:dt] = dt
    end

    solve_kwargs[:progress] = cache.progress

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

function solve_dae_mass_matrix(cache, dt, maxit, u0, p)
    if cache.f.cons === nothing
        error("DAEOptimizer requires constraints. Please provide a function with `cons` defined.")
    end
    n = length(u0)
    m = length(cache.ucons)

    if m > n
        error("DAEOptimizer with mass matrix method requires the number of constraints to be less than or equal to the number of variables.")
    end
    M = Diagonal([ones(n-m); zeros(m)])
    function f_mass!(du, u, p_, t)
        cache.f.grad(du, u, p)
        @. du = -du
        consout = @view du[(n-m)+1:end]
        cache.f.cons(consout, u)
        return nothing
    end

    ss_prob = SteadyStateProblem(ODEFunction(f_mass!, mass_matrix = M), u0, p)

    if cache.callback !== Optimization.DEFAULT_CALLBACK
        condition = (u, t, integrator) -> true
        affect! = (integrator) -> begin
            u_opt = integrator.u isa AbstractArray ? integrator.u : integrator.u.u
            l = cache.f(integrator.u, integrator.p)
            cache.callback(integrator.u, l)
        end
        cb = DiscreteCallback(condition, affect!)
        solve_kwargs = Dict{Symbol, Any}(:callback => cb)
    else
        solve_kwargs = Dict{Symbol, Any}()
    end
    
    solve_kwargs[:progress] = cache.progress
    if maxit !== nothing; solve_kwargs[:maxiters] = maxit; end
    if dt   !== nothing; solve_kwargs[:dt] = dt; end

    sol = solve(ss_prob, DynamicSS(cache.opt.solver); solve_kwargs...)
    # if sol.retcode ≠ ReturnCode.Success
    #     # you may still accept Default or warn
    # end
    u_ext = sol.u  
    u_final = u_ext[1:n]
    return SciMLBase.build_solution(cache, cache.opt, u_final, cache.f(u_final, p);
        retcode = sol.retcode)
end

function solve_dae_implicit(cache, dt, maxit, u0, p)
    if cache.f.cons === nothing
        error("DAEOptimizer requires constraints. Please provide a function with `cons` defined.")
    end

    n = length(u0)
    m = length(cache.ucons)

    if m > n
        error("DAEOptimizer with mass matrix method requires the number of constraints to be less than or equal to the number of variables.")
    end

    function dae_residual!(res, du, u, p_, t)
        cache.f.grad(res, u, p)
        @. res = du-res
        consout = @view res[(n-m)+1:end]
        cache.f.cons(consout, u)
        return nothing
    end

    tspan = (0.0, 10.0)
    du0 = zero(u0)
    prob = DAEProblem(dae_residual!, du0, u0, tspan, p)

    if cache.callback !== Optimization.DEFAULT_CALLBACK
        condition = (u, t, integrator) -> true
        affect! = (integrator) -> begin
            u_opt = integrator.u isa AbstractArray ? integrator.u : integrator.u.u
            l = cache.f(integrator.u, integrator.p)
            cache.callback(integrator.u, l)
        end
        cb = DiscreteCallback(condition, affect!)
        solve_kwargs = Dict{Symbol, Any}(:callback => cb)
    else
        solve_kwargs = Dict{Symbol, Any}()
    end
    
    solve_kwargs[:progress] = cache.progress

    if maxit !== nothing; solve_kwargs[:maxiters] = maxit; end
    if dt !== nothing; solve_kwargs[:dt] = dt; end
    solve_kwargs[:initializealg] = ShampineCollocationInit()

    sol = solve(prob, cache.opt.solver; solve_kwargs...)
    u_ext = sol.u
    u_final = u_ext[end][1:n]

    return SciMLBase.build_solution(cache, cache.opt, u_final, cache.f(u_final, p);
        retcode = sol.retcode)
end


end 
