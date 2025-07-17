module OptimizationOptimJL

using Reexport
@reexport using Optim, Optimization
using Optimization.SciMLBase, SparseArrays
decompose_trace(trace::Optim.OptimizationTrace) = last(trace)
decompose_trace(trace::Optim.OptimizationState) = trace

SciMLBase.allowsconstraints(::IPNewton) = true
SciMLBase.allowsbounds(opt::Optim.AbstractOptimizer) = true
SciMLBase.allowsbounds(opt::Optim.SimulatedAnnealing) = false
SciMLBase.requiresbounds(opt::Optim.Fminbox) = true
SciMLBase.requiresbounds(opt::Optim.SAMIN) = true
SciMLBase.supports_opt_cache_interface(opt::Optim.AbstractOptimizer) = true
SciMLBase.supports_opt_cache_interface(opt::Union{Optim.Fminbox, Optim.SAMIN}) = true
SciMLBase.supports_opt_cache_interface(opt::Optim.ConstrainedOptimizer) = true
function SciMLBase.requiresgradient(opt::Optim.AbstractOptimizer)
    !(opt isa Optim.ZerothOrderOptimizer)
end
SciMLBase.requiresgradient(::IPNewton) = true
SciMLBase.requireshessian(::IPNewton) = true
SciMLBase.requiresconsjac(::IPNewton) = true
SciMLBase.requiresconshess(::IPNewton) = true
function SciMLBase.requireshessian(opt::Union{
        Optim.Newton, Optim.NewtonTrustRegion, Optim.KrylovTrustRegion})
    true
end
SciMLBase.requiresgradient(opt::Optim.Fminbox) = true
# SciMLBase.allowsfg(opt::Union{Optim.AbstractOptimizer, Optim.ConstrainedOptimizer, Optim.Fminbox, Optim.SAMIN}) = true

function __map_optimizer_args(cache::OptimizationCache,
        opt::Union{Optim.AbstractOptimizer, Optim.Fminbox,
            Optim.SAMIN, Optim.ConstrainedOptimizer};
        callback = nothing,
        maxiters::Union{Number, Nothing} = nothing,
        local_maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        kwargs...)
    mapped_args = (; extended_trace = true, kwargs...)

    if !isnothing(abstol)
        mapped_args = (; mapped_args..., f_abstol = abstol)
    end

    if !isnothing(callback)
        mapped_args = (; mapped_args..., callback = callback)
    end

    if !isnothing(maxiters)
        if opt isa Optim.Fminbox
            if !isnothing(local_maxiters)
                mapped_args = (;
                    mapped_args...,
                    outer_iterations = maxiters,
                    iterations = local_maxiters)
            else
                mapped_args = (; mapped_args..., outer_iterations = maxiters)
            end
        else
            mapped_args = (; mapped_args..., iterations = maxiters)
        end
    end

    if !isnothing(local_maxiters) && opt isa Optim.Fminbox
        mapped_args = (; mapped_args..., iterations = local_maxiters)
    end

    if !isnothing(maxtime)
        mapped_args = (; mapped_args..., time_limit = maxtime)
    end

    if !isnothing(reltol)
        mapped_args = (; mapped_args..., f_reltol = reltol)
    end

    return Optim.Options(; mapped_args...)
end

function SciMLBase.__init(prob::OptimizationProblem,
        opt::Union{Optim.AbstractOptimizer, Optim.Fminbox,
            Optim.SAMIN, Optim.ConstrainedOptimizer
        };
        callback = (args...) -> (false),
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        progress = false,
        kwargs...)
    if !isnothing(prob.lb) || !isnothing(prob.ub)
        if !(opt isa Union{Optim.Fminbox, Optim.SAMIN, Optim.AbstractConstrainedOptimizer})
            if opt isa Optim.ParticleSwarm
                opt = Optim.ParticleSwarm(; lower = prob.lb, upper = prob.ub,
                    n_particles = opt.n_particles)
            else
                if prob.f isa OptimizationFunction &&
                   (!(prob.f.adtype isa SciMLBase.NoAD) || !isnothing(prob.f.grad))
                    opt = Optim.Fminbox(opt)
                else
                    throw(ArgumentError("Fminbox($opt) requires gradients, use `OptimizationFunction` either with a valid AD backend https://docs.sciml.ai/Optimization/stable/API/ad/ or a provided 'grad' function."))
                end
            end
        end
    end

    maxiters = Optimization._check_and_convert_maxiters(maxiters)
    maxtime = Optimization._check_and_convert_maxtime(maxtime)
    return OptimizationCache(prob, opt; callback, maxiters, maxtime, abstol,
        reltol, progress,
        kwargs...)
end

function SciMLBase.__solve(cache::OptimizationCache{
        F,
        RC,
        LB,
        UB,
        LC,
        UC,
        S,
        O,
        D,
        P
}) where {
        F,
        RC,
        LB,
        UB, LC, UC,
        S,
        O <:
        Optim.AbstractOptimizer,
        D,
        P
}
    local x, cur, state
    !(cache.opt isa Optim.ZerothOrderOptimizer) && cache.f.grad === nothing &&
        error("Use OptimizationFunction to pass the derivatives or automatically generate them with one of the autodiff backends")

    function _cb(trace)
        metadata = decompose_trace(trace).metadata
        θ = metadata[cache.opt isa Optim.NelderMead ? "centroid" : "x"]
        opt_state = Optimization.OptimizationState(iter = trace.iteration,
            u = θ,
            objective = trace.value,
            grad = get(metadata, "g(x)", nothing),
            hess = get(metadata, "h(x)", nothing),
            original = trace)
        cb_call = cache.callback(opt_state, trace.value)
        if !(cb_call isa Bool)
            error("The callback should return a boolean `halt` for whether to stop the optimization process.")
        end
        cb_call
    end

    _loss = function (θ)
        x = cache.f.f(θ, cache.p)
        __x = first(x)
        return cache.sense === Optimization.MaxSense ? -__x : __x
    end

    if cache.f.fg === nothing
        fg! = function (G, θ)
            if G !== nothing
                cache.f.grad(G, θ)
                if cache.sense === Optimization.MaxSense
                    G .*= -one(eltype(G))
                end
            end
            return _loss(θ)
        end
    else
        fg! = cache.f.fg
    end

    if cache.opt isa Optim.KrylovTrustRegion
        hv = function (H, θ, v)
            cache.f.hv(H, θ, v)
            if cache.sense === Optimization.MaxSense
                H .*= -one(eltype(H))
            end
        end
        optim_f = Optim.TwiceDifferentiableHV(_loss, fg!, hv, cache.u0)
    else
        gg = function (G, θ)
            cache.f.grad(G, θ)
            if cache.sense === Optimization.MaxSense
                G .*= -one(eltype(G))
            end
        end

        hh = function (H, θ)
            cache.f.hess(H, θ)
            if cache.sense === Optimization.MaxSense
                H .*= -one(eltype(H))
            end
        end
        u0_type = eltype(cache.u0)
        optim_f = Optim.TwiceDifferentiable(_loss, gg, fg!, hh, cache.u0,
            real(zero(u0_type)),
            Optim.NLSolversBase.alloc_DF(cache.u0,
                real(zero(u0_type))),
            isnothing(cache.f.hess_prototype) ?
            Optim.NLSolversBase.alloc_H(cache.u0,
                real(zero(u0_type))) :
            convert.(u0_type, cache.f.hess_prototype))
    end

    opt_args = __map_optimizer_args(cache, cache.opt, callback = _cb,
        maxiters = cache.solver_args.maxiters,
        maxtime = cache.solver_args.maxtime,
        abstol = cache.solver_args.abstol,
        reltol = cache.solver_args.reltol;
        cache.solver_args...)

    t0 = time()
    opt_res = Optim.optimize(optim_f, cache.u0, cache.opt, opt_args)
    t1 = time()
    opt_ret = Symbol(Optim.converged(opt_res))
    stats = Optimization.OptimizationStats(; iterations = opt_res.iterations,
        time = t1 - t0, fevals = opt_res.f_calls, gevals = opt_res.g_calls,
        hevals = opt_res.h_calls)
    SciMLBase.build_solution(cache, cache.opt,
        opt_res.minimizer,
        cache.sense === Optimization.MaxSense ? -opt_res.minimum :
        opt_res.minimum; original = opt_res, retcode = opt_ret,
        stats = stats)
end

function SciMLBase.__solve(cache::OptimizationCache{
        F,
        RC,
        LB,
        UB,
        LC,
        UC,
        S,
        O,
        D,
        P
}) where {
        F,
        RC,
        LB,
        UB, LC, UC,
        S,
        O <:
        Union{
            Optim.Fminbox,
            Optim.SAMIN
        },
        D,
        P
}
    local x, cur, state

    function _cb(trace)
        metadata = decompose_trace(trace).metadata
        θ = !(cache.opt isa Optim.SAMIN) && cache.opt.method == Optim.NelderMead() ?
            metadata["centroid"] :
            metadata["x"]
        opt_state = Optimization.OptimizationState(iter = trace.iteration,
            u = θ,
            objective = trace.value,
            grad = get(metadata, "g(x)", nothing),
            hess = get(metadata, "h(x)", nothing),
            original = trace)
        cb_call = cache.callback(opt_state, trace.value)
        if !(cb_call isa Bool)
            error("The callback should return a boolean `halt` for whether to stop the optimization process.")
        end
        cb_call
    end

    _loss = function (θ)
        x = cache.f.f(θ, cache.p)
        __x = first(x)
        return cache.sense === Optimization.MaxSense ? -__x : __x
    end

    if cache.f.fg === nothing
        fg! = function (G, θ)
            if G !== nothing
                cache.f.grad(G, θ)
                if cache.sense === Optimization.MaxSense
                    G .*= -one(eltype(G))
                end
            end
            return _loss(θ)
        end
    else
        fg! = cache.f.fg
    end

    gg = function (G, θ)
        cache.f.grad(G, θ)
        if cache.sense === Optimization.MaxSense
            G .*= -one(eltype(G))
        end
    end
    optim_f = Optim.OnceDifferentiable(_loss, gg, fg!, cache.u0)

    opt_args = __map_optimizer_args(cache, cache.opt, callback = _cb,
        maxiters = cache.solver_args.maxiters,
        maxtime = cache.solver_args.maxtime,
        abstol = cache.solver_args.abstol,
        reltol = cache.solver_args.reltol;
        cache.solver_args...)

    t0 = time()
    opt_res = Optim.optimize(optim_f, cache.lb, cache.ub, cache.u0, cache.opt, opt_args)
    t1 = time()
    opt_ret = Symbol(Optim.converged(opt_res))
    stats = Optimization.OptimizationStats(; iterations = opt_res.iterations,
        time = t1 - t0, fevals = opt_res.f_calls, gevals = opt_res.g_calls,
        hevals = opt_res.h_calls)
    SciMLBase.build_solution(cache, cache.opt,
        opt_res.minimizer, opt_res.minimum;
        original = opt_res, retcode = opt_ret, stats = stats)
end

function SciMLBase.__solve(cache::OptimizationCache{
        F,
        RC,
        LB,
        UB,
        LC,
        UC,
        S,
        O,
        D,
        P
}) where {
        F,
        RC,
        LB,
        UB, LC, UC,
        S,
        O <:
        Optim.ConstrainedOptimizer,
        D,
        P
}
    local x, cur, state

    function _cb(trace)
        metadata = decompose_trace(trace).metadata
        opt_state = Optimization.OptimizationState(iter = trace.iteration,
            u = metadata["x"],
            grad = get(metadata, "g(x)", nothing),
            hess = get(metadata, "h(x)", nothing),
            objective = trace.value,
            original = trace)
        cb_call = cache.callback(opt_state, trace.value)
        if !(cb_call isa Bool)
            error("The callback should return a boolean `halt` for whether to stop the optimization process.")
        end
        cb_call
    end

    _loss = function (θ)
        x = cache.f.f(θ, cache.p)
        __x = first(x)
        return cache.sense === Optimization.MaxSense ? -__x : __x
    end

    if cache.f.fg === nothing
        fg! = function (G, θ)
            if G !== nothing
                cache.f.grad(G, θ)
                if cache.sense === Optimization.MaxSense
                    G .*= -one(eltype(G))
                end
            end
            return _loss(θ)
        end
    else
        fg! = cache.f.fg
    end

    gg = function (G, θ)
        cache.f.grad(G, θ)
        if cache.sense === Optimization.MaxSense
            G .*= -one(eltype(G))
        end
    end

    hh = function (H, θ)
        cache.f.hess(H, θ)
        if cache.sense === Optimization.MaxSense
            H .*= -one(eltype(H))
        end
    end
    u0_type = eltype(cache.u0)

    optim_f = if SciMLBase.requireshessian(cache.opt)
        Optim.TwiceDifferentiable(_loss, gg, fg!, hh, cache.u0,
            real(zero(u0_type)),
            Optim.NLSolversBase.alloc_DF(cache.u0,
                real(zero(u0_type))),
            isnothing(cache.f.hess_prototype) ?
            Optim.NLSolversBase.alloc_H(cache.u0,
                real(zero(u0_type))) :
            convert.(u0_type, cache.f.hess_prototype))
    else
        Optim.OnceDifferentiable(_loss, gg, fg!, cache.u0, 
                        real(zero(u0_type)),
                        Optim.NLSolversBase.alloc_DF(cache.u0,
                            real(zero(u0_type))))
    end

    cons_hl! = function (h, θ, λ)
        res = [similar(h) for i in 1:length(λ)]
        cache.f.cons_h(res, θ)
        for i in 1:length(λ)
            h .+= λ[i] * res[i]
        end
    end

    lb = cache.lb === nothing ? [] : cache.lb
    ub = cache.ub === nothing ? [] : cache.ub

    optim_fc = if SciMLBase.requireshessian(cache.opt)
        if cache.f.cons !== nothing
            Optim.TwiceDifferentiableConstraints(cache.f.cons, cache.f.cons_j,
                cons_hl!,
                lb, ub,
                cache.lcons, cache.ucons)
        else
            Optim.TwiceDifferentiableConstraints(lb, ub)
        end
    else
        if cache.f.cons !== nothing
            Optim.OnceDifferentiableConstraints(cache.f.cons, cache.f.cons_j,
                lb, ub,
                cache.lcons, cache.ucons)
        else
            Optim.OnceDifferentiableConstraints(lb, ub)
        end
    end
    
    opt_args = __map_optimizer_args(cache, cache.opt, callback = _cb,
        maxiters = cache.solver_args.maxiters,
        maxtime = cache.solver_args.maxtime,
        abstol = cache.solver_args.abstol,
        reltol = cache.solver_args.reltol;
        cache.solver_args...)

    t0 = time()
    if lb === nothing && ub === nothing && cache.f.cons === nothing
        opt_res = Optim.optimize(optim_f, cache.u0, cache.opt, opt_args)
    else
        opt_res = Optim.optimize(optim_f, optim_fc, cache.u0, cache.opt, opt_args)
    end
    t1 = time()
    opt_ret = Symbol(Optim.converged(opt_res))
    stats = Optimization.OptimizationStats(; iterations = opt_res.iterations,
        time = t1 - t0, fevals = opt_res.f_calls, gevals = opt_res.g_calls,
        hevals = opt_res.h_calls)
    SciMLBase.build_solution(cache, cache.opt,
        opt_res.minimizer, opt_res.minimum;
        original = opt_res, retcode = opt_ret,
        stats = stats)
end

using PrecompileTools
PrecompileTools.@compile_workload begin
    function obj_f(x, p)
        A = p[1]
        b = p[2]
        return sum((A * x .- b) .^ 2)
    end

    function solve_nonnegative_least_squares(A, b, solver)
        optf = Optimization.OptimizationFunction(obj_f, Optimization.AutoForwardDiff())
        prob = Optimization.OptimizationProblem(optf, ones(size(A, 2)), (A, b),
            lb = zeros(size(A, 2)), ub = Inf * ones(size(A, 2)))
        x = OptimizationOptimJL.solve(prob, solver, maxiters = 5000, maxtime = 100)

        return x
    end

    solver_list = [OptimizationOptimJL.LBFGS(),
        OptimizationOptimJL.ConjugateGradient(),
        OptimizationOptimJL.GradientDescent(),
        OptimizationOptimJL.BFGS()]

    for solver in solver_list
        x = solve_nonnegative_least_squares(rand(4, 4), rand(4), solver)
        x = solve_nonnegative_least_squares(rand(35, 35), rand(35), solver)
        x = solve_nonnegative_least_squares(rand(35, 10), rand(35), solver)
    end
end

end
