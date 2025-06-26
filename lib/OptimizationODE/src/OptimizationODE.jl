module OptimizationODE

using Reexport
@reexport using Optimization, SciMLBase
using LinearAlgebra, ForwardDiff

using NonlinearSolve
using OrdinaryDiffEq, DifferentialEquations, SteadyStateDiffEq, Sundials

export ODEOptimizer, ODEGradientDescent, RKChebyshevDescent, RKAccelerated, HighOrderDescent
export DAEOptimizer, DAEMassMatrix, DAEIndexing

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

DAEMassMatrix() = DAEOptimizer(Rodas5())
DAEIndexing() = DAEOptimizer(IDA())


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
    maxiters=nothing, differential_vars=nothing, kwargs...)
    return OptimizationCache(prob, opt; callback=callback, progress=progress, dt=dt,
        maxiters=maxiters, differential_vars=differential_vars, kwargs...)
end


function get_solver_type(opt::DAEOptimizer)
    if opt.solver isa Union{Rodas5, RadauIIA5, ImplicitEuler, Trapezoid}
        return :mass_matrix
    else
        return :indexing
    end
end


function handle_parameters(p)
    if p isa SciMLBase.NullParameters
        return Float64[]
    else
        return p
    end
end

function setup_progress_callback(cache, solve_kwargs)
    if get(cache.solver_args, :progress, false)
        condition = (u, t, integrator) -> true
        affect! = (integrator) -> begin
            u_opt = integrator.u isa AbstractArray ? integrator.u : integrator.u.u
            cache.solver_args[:callback](u_opt, integrator.p, integrator.t)
        end
        cb = DiscreteCallback(condition, affect!)
        solve_kwargs[:callback] = cb
    end
    return solve_kwargs
end


function SciMLBase.__solve(
    cache::OptimizationCache{F,RC,LB,UB,LC,UC,S,O,D,P,C}
    ) where {F,RC,LB,UB,LC,UC,S,O<:Union{ODEOptimizer,DAEOptimizer},D,P,C}

    dt = get(cache.solver_args, :dt, nothing)
    maxit = get(cache.solver_args, :maxiters, nothing)
    differential_vars = get(cache.solver_args, :differential_vars, nothing)
    u0 = copy(cache.u0)
    p = handle_parameters(cache.p)  # Properly handle NullParameters

    if cache.opt isa ODEOptimizer
        return solve_ode(cache, dt, maxit, u0, p)
    else
        solver_method = get_solver_type(cache.opt)
        if solver_method == :mass_matrix
            return solve_dae_mass_matrix(cache, dt, maxit, u0, p)
        else
            return solve_dae_indexing(cache, dt, maxit, u0, p, differential_vars)
        end
    end
end

function solve_ode(cache, dt, maxit, u0, p)
    if cache.f.grad === nothing
        error("ODEOptimizer requires a gradient. Please provide a function with `grad` defined.")
    end

    function f!(du, u, p, t)
        grad_vec = similar(u)
        if isempty(p)
            cache.f.grad(grad_vec, u)
        else
            cache.f.grad(grad_vec, u, p)
        end
        @. du = -grad_vec
        return nothing
    end

    ss_prob = SteadyStateProblem(f!, u0, p)

    algorithm = DynamicSS(cache.opt.solver)

    cb = cache.callback
    if cb != Optimization.DEFAULT_CALLBACK || get(cache.solver_args,:progress,false)
        function condition(u, t, integrator) true end
        function affect!(integrator)
            u_now = integrator.u
            cache.callback(u_now, integrator.p, integrator.t)
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

function solve_dae_mass_matrix(cache, dt, maxit, u0, p)
    if cache.f.cons === nothing
        return solve_ode(cache, dt, maxit, u0, p)
    end
    x=u0
    cons_vals = cache.f.cons(x, p)
    n = length(u0)
    m = length(cons_vals)
    u0_extended = vcat(u0, zeros(m))
    M = Diagonal(ones(n + m))


    function f_mass!(du, u, p_, t)
        x = @view u[1:n]
        λ = @view u[n+1:end]
        grad_f = similar(x)
        if cache.f.grad !== nothing
            cache.f.grad(grad_f, x, p_)
        else
            grad_f .= ForwardDiff.gradient(z -> cache.f.f(z, p_), x)
        end
        J = Matrix{eltype(x)}(undef, m, n)
        cache.f.cons_j !== nothing && cache.f.cons_j(J, x)

        @. du[1:n] = -grad_f - (J' * λ)
        consv = cache.f.cons(x, p_)
        @. du[n+1:end] = consv
        return nothing
    end

    if m == 0
        optf = ODEFunction(f_mass!, mass_matrix = I(n))
        prob = ODEProblem(optf, u0, (0.0, 1.0), p)
        return solve(prob, HighOrderDescent(); dt=dt, maxiters=maxit)
    end

    ss_prob = SteadyStateProblem(ODEFunction(f_mass!, mass_matrix = M), u0_extended, p)

    solve_kwargs = setup_progress_callback(cache, Dict())
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


function solve_dae_indexing(cache, dt, maxit, u0, p, differential_vars)
    if cache.f.cons === nothing
        return solve_ode(cache, dt, maxit, u0, p)
    end
    x=u0
    cons_vals = cache.f.cons(x, p)
    n = length(u0)
    m = length(cons_vals)
    u0_ext = vcat(u0, zeros(m))
    du0_ext = zeros(n + m)

    if differential_vars === nothing
        differential_vars = vcat(fill(true, n), fill(false, m))
    else
        if length(differential_vars) == n
            differential_vars = vcat(differential_vars, fill(false, m))
        elseif length(differential_vars) == n + m
            # use as is
        else
            error("differential_vars length must be number of variables ($n) or extended size ($(n+m))")
        end
    end

    function dae_residual!(res, du, u, p_, t)
        x = @view u[1:n]
        λ = @view u[n+1:end]
        du_x = @view du[1:n]
        grad_f = similar(x)
        cache.f.grad(grad_f, x, p_)
        J = zeros(m, n)
        cache.f.cons_j !== nothing && cache.f.cons_j(J, x)

        @. res[1:n] = du_x + grad_f + J' * λ
        consv = cache.f.cons(x, p_)
        @. res[n+1:end] = consv
        return nothing
    end

    if m == 0
        optf = ODEFunction(dae_residual!, differential_vars = differential_vars)
        prob = ODEProblem(optf, du0_ext, (0.0, 1.0), p)
        return solve(prob, HighOrderDescent(); dt=dt, maxiters=maxit)
    end

    tspan = (0.0, 10.0)
    prob = DAEProblem(dae_residual!, du0_ext, u0_ext, tspan, p;
                      differential_vars = differential_vars)

    solve_kwargs = setup_progress_callback(cache, Dict())
    if maxit !== nothing; solve_kwargs[:maxiters] = maxit; end
    if dt !== nothing; solve_kwargs[:dt] = dt; end
    if hasfield(typeof(cache.opt.solver), :initializealg)
        solve_kwargs[:initializealg] = BrownFullBasicInit()
    end

    sol = solve(prob, cache.opt.solver; solve_kwargs...)
    u_ext = sol.u
    u_final = u_ext[end][1:n]

    return SciMLBase.build_solution(cache, cache.opt, u_final, cache.f(u_final, p);
        retcode = sol.retcode)
end


end 
