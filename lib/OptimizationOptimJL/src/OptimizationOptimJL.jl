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
    if !isnothing(abstol)
        @warn "common abstol is currently not used by $(opt)"
    end

    mapped_args = (; extended_trace = true, kwargs...)

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
        mapped_args = (; mapped_args..., f_tol = reltol)
    end

    return Optim.Options(; mapped_args...)
end

function SciMLBase.__init(prob::OptimizationProblem,
        opt::Union{Optim.AbstractOptimizer, Optim.Fminbox,
            Optim.SAMIN, Optim.ConstrainedOptimizer
        },
        data = Optimization.DEFAULT_DATA;
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
            elseif opt isa Optim.SimulatedAnnealing
                @warn "$(opt) can currently not be wrapped in Fminbox(). The lower and upper bounds thus will be ignored. Consider using a different optimizer or open an issue with Optim.jl"
            else
                opt = Optim.Fminbox(opt)
            end
        end
    end

    maxiters = if data != Optimization.DEFAULT_DATA
        length(data)
    else
        maxiters
    end

    maxiters = Optimization._check_and_convert_maxiters(maxiters)
    maxtime = Optimization._check_and_convert_maxtime(maxtime)
    return OptimizationCache(prob, opt, data; callback, maxiters, maxtime, abstol,
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

    cur, state = iterate(cache.data)

    !(cache.opt isa Optim.ZerothOrderOptimizer) && cache.f.grad === nothing &&
        error("Use OptimizationFunction to pass the derivatives or automatically generate them with one of the autodiff backends")

    function _cb(trace)
        metadata = decompose_trace(trace).metadata
        θ = metadata[cache.opt isa Optim.NelderMead ? "centroid" : "x"]
        opt_state = Optimization.OptimizationState(iter = trace.iteration,
            u = θ,
            objective = x[1],
            grad = get(metadata, "g(x)", nothing),
            hess = get(metadata, "h(x)", nothing),
            original = trace)
        cb_call = cache.callback(opt_state, x...)
        if !(cb_call isa Bool)
            error("The callback should return a boolean `halt` for whether to stop the optimization process.")
        end
        nx_itr = iterate(cache.data, state)
        if isnothing(nx_itr)
            true
        else
            cur, state = nx_itr
            cb_call
        end
    end

    _loss = function (θ)
        x = cache.f.f(θ, cache.p, cur...)
        __x = first(x)
        return cache.sense === Optimization.MaxSense ? -__x : __x
    end

    fg! = function (G, θ)
        if G !== nothing
            cache.f.grad(G, θ, cur...)
            if cache.sense === Optimization.MaxSense
                G .*= -one(eltype(G))
            end
        end
        return _loss(θ)
    end

    if cache.opt isa Optim.KrylovTrustRegion
        hv = function (H, θ, v)
            cache.f.hv(H, θ, v, cur...)
            if cache.sense === Optimization.MaxSense
                H .*= -one(eltype(H))
            end
        end
        optim_f = Optim.TwiceDifferentiableHV(_loss, fg!, hv, cache.u0)
    else
        gg = function (G, θ)
            cache.f.grad(G, θ, cur...)
            if cache.sense === Optimization.MaxSense
                G .*= -one(eltype(G))
            end
        end

        hh = function (H, θ)
            cache.f.hess(H, θ, cur...)
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

    cur, state = iterate(cache.data)

    function _cb(trace)
        metadata = decompose_trace(trace).metadata
        θ = !(cache.opt isa Optim.SAMIN) && cache.opt.method == Optim.NelderMead() ?
            metadata["centroid"] :
            metadata["x"]
        opt_state = Optimization.OptimizationState(iter = trace.iteration,
            u = θ,
            objective = x[1],
            grad = get(metadata, "g(x)", nothing),
            hess = get(metadata, "h(x)", nothing),
            original = trace)
        cb_call = cache.callback(opt_state, x...)
        if !(cb_call isa Bool)
            error("The callback should return a boolean `halt` for whether to stop the optimization process.")
        end
        nx_itr = iterate(cache.data, state)
        if isnothing(nx_itr)
            true
        else
            cur, state = nx_itr
            cb_call
        end
    end

    _loss = function (θ)
        x = cache.f.f(θ, cache.p, cur...)
        __x = first(x)
        return cache.sense === Optimization.MaxSense ? -__x : __x
    end
    fg! = function (G, θ)
        if G !== nothing
            cache.f.grad(G, θ, cur...)
            if cache.sense === Optimization.MaxSense
                G .*= -one(eltype(G))
            end
        end
        return _loss(θ)
    end

    gg = function (G, θ)
        cache.f.grad(G, θ, cur...)
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

    cur, state = iterate(cache.data)

    function _cb(trace)
        metadata = decompose_trace(trace).metadata
        opt_state = Optimization.OptimizationState(iter = trace.iteration,
            u = metadata["x"],
            grad = get(metadata, "g(x)", nothing),
            hess = get(metadata, "h(x)", nothing),
            objective = x[1],
            original = trace)
        cb_call = cache.callback(opt_state, x...)
        if !(cb_call isa Bool)
            error("The callback should return a boolean `halt` for whether to stop the optimization process.")
        end
        nx_itr = iterate(cache.data, state)
        if isnothing(nx_itr)
            true
        else
            cur, state = nx_itr
            cb_call
        end
    end

    _loss = function (θ)
        x = cache.f.f(θ, cache.p, cur...)
        __x = first(x)
        return cache.sense === Optimization.MaxSense ? -__x : __x
    end
    fg! = function (G, θ)
        if G !== nothing
            cache.f.grad(G, θ, cur...)
            if cache.sense === Optimization.MaxSense
                G .*= -one(eltype(G))
            end
        end
        return _loss(θ)
    end
    gg = function (G, θ)
        cache.f.grad(G, θ, cur...)
        if cache.sense === Optimization.MaxSense
            G .*= -one(eltype(G))
        end
    end

    hh = function (H, θ)
        cache.f.hess(H, θ, cur...)
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

    cons_hl! = function (h, θ, λ)
        res = [similar(h) for i in 1:length(λ)]
        cache.f.cons_h(res, θ)
        for i in 1:length(λ)
            h .+= λ[i] * res[i]
        end
    end

    lb = cache.lb === nothing ? [] : cache.lb
    ub = cache.ub === nothing ? [] : cache.ub
    if cache.f.cons !== nothing
        optim_fc = Optim.TwiceDifferentiableConstraints(cache.f.cons, cache.f.cons_j,
            cons_hl!,
            lb, ub,
            cache.lcons, cache.ucons)
    else
        optim_fc = Optim.TwiceDifferentiableConstraints(lb, ub)
    end

    opt_args = __map_optimizer_args(cache, cache.opt, callback = _cb,
        maxiters = cache.solver_args.maxiters,
        maxtime = cache.solver_args.maxtime,
        abstol = cache.solver_args.abstol,
        reltol = cache.solver_args.reltol;
        cache.solver_args...)

    t0 = time()
    opt_res = Optim.optimize(optim_f, optim_fc, cache.u0, cache.opt, opt_args)
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

end
