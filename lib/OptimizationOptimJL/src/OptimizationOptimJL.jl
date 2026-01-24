module OptimizationOptimJL

using Reexport
@reexport using Optim, OptimizationBase
using SciMLBase, SparseArrays
decompose_trace(trace::Optim.OptimizationTrace) = last(trace)
decompose_trace(trace::Optim.OptimizationState) = trace

# Helper to extract callback information from both Optim v1 (trace-based) and v2 (state-based) callbacks.
# In Optim v1, callbacks receive an OptimizationTrace or OptimizationState (trace entry) with a metadata dict.
# In Optim v2, callbacks receive the optimizer state directly (e.g., BFGSState, NelderMeadState).
function _extract_callback_state(state_or_trace, opt, iter_counter)
    if state_or_trace isa Optim.OptimizationTrace || state_or_trace isa Optim.OptimizationState
        # Optim v1 path
        trace_state = decompose_trace(state_or_trace)
        metadata = trace_state.metadata
        if opt isa Optim.NelderMead
            θ = metadata["centroid"]
        else
            θ = metadata["x"]
        end
        loss_val = SciMLBase.value(trace_state.value)
        iter = trace_state.iteration
        g = get(metadata, "g(x)", nothing)
        h = get(metadata, "h(x)", nothing)
    else
        # Optim v2 path - state is an AbstractOptimizerState subtype
        if opt isa Optim.NelderMead && hasproperty(state_or_trace, :x_centroid)
            θ = state_or_trace.x_centroid
        else
            θ = state_or_trace.x
        end
        loss_val = SciMLBase.value(state_or_trace.f_x)
        iter = iter_counter[]
        g = hasproperty(state_or_trace, :g_x) ? state_or_trace.g_x : nothing
        h = hasproperty(state_or_trace, :H_x) ? state_or_trace.H_x : nothing
    end
    return (; θ, loss_val, iter, g, h, original = state_or_trace)
end

# Helper to construct an objective with Hessian-vector product support.
# In Optim v1, TwiceDifferentiableHV was a separate type.
# In Optim v2, TwiceDifferentiable gained an `hv` field directly.
function _make_hv_objective(f, fg!, hv!, x0)
    if isdefined(Optim, :TwiceDifferentiableHV)
        # Optim v1 path
        return Optim.TwiceDifferentiableHV(f, fg!, hv!, x0)
    else
        # Optim v2 path - construct TwiceDifferentiable with hv field
        NLB = Optim.NLSolversBase
        u0_type = eltype(x0)
        F = real(zero(u0_type))
        G = NLB.alloc_DF(x0, F)
        H = NLB.alloc_H(x0, F)
        g! = let fg! = fg!
            function (_g, _x)
                fg!(_g, _x)
                return nothing
            end
        end
        h! = nothing
        dfh! = nothing
        fdfh! = nothing
        x_f = fill!(similar(x0, u0_type), NaN)
        x_df = fill!(similar(x0, u0_type), NaN)
        x_jvp = fill!(similar(x0, u0_type), NaN)
        v_jvp = fill!(similar(x0, u0_type), NaN)
        x_h = fill!(similar(x0, u0_type), NaN)
        x_hvp = fill!(similar(x0, u0_type), NaN)
        v_hvp = fill!(similar(x0, u0_type), NaN)
        HVP = copy(G)
        JVP = F
        return NLB.TwiceDifferentiable(
            f, g!, fg!, nothing, nothing, dfh!, fdfh!, h!, hv!,
            F, copy(G), JVP, copy(H), HVP,
            x_f, x_df, x_jvp, v_jvp, x_h, x_hvp, v_hvp,
            0, 0, 0, 0, 0
        )
    end
end

SciMLBase.allowsconstraints(::IPNewton) = true
SciMLBase.allowsbounds(opt::Optim.AbstractOptimizer) = true
SciMLBase.allowsbounds(opt::Optim.SimulatedAnnealing) = false
SciMLBase.requiresbounds(opt::Optim.Fminbox) = true
SciMLBase.requiresbounds(opt::Optim.SAMIN) = true

SciMLBase.has_init(opt::Optim.AbstractOptimizer) = true
SciMLBase.has_init(opt::Union{Optim.Fminbox, Optim.SAMIN}) = true
SciMLBase.has_init(opt::Optim.ConstrainedOptimizer) = true

SciMLBase.allowscallback(opt::Optim.AbstractOptimizer) = true
SciMLBase.allowscallback(opt::Union{Optim.Fminbox, Optim.SAMIN}) = true
SciMLBase.allowscallback(opt::Optim.ConstrainedOptimizer) = true

function SciMLBase.requiresgradient(opt::Optim.AbstractOptimizer)
    return !(opt isa Optim.ZerothOrderOptimizer)
end
SciMLBase.requiresgradient(::IPNewton) = true
SciMLBase.requireshessian(::IPNewton) = true
SciMLBase.requiresconsjac(::IPNewton) = true
SciMLBase.requiresconshess(::IPNewton) = true
function SciMLBase.requireshessian(
        opt::Union{
            Optim.Newton, Optim.NewtonTrustRegion, Optim.KrylovTrustRegion,
        }
    )
    return true
end
SciMLBase.requiresgradient(opt::Optim.Fminbox) = true
# SciMLBase.allowsfg(opt::Union{Optim.AbstractOptimizer, Optim.ConstrainedOptimizer, Optim.Fminbox, Optim.SAMIN}) = true

function __map_optimizer_args(
        cache::OptimizationBase.OptimizationCache,
        opt::Union{
            Optim.AbstractOptimizer, Optim.Fminbox,
            Optim.SAMIN, Optim.ConstrainedOptimizer,
        };
        callback = nothing,
        maxiters::Union{Number, Nothing} = nothing,
        local_maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        verbose = false,
        kwargs...
    )
    mapped_args = (; extended_trace = true, show_trace = verbose, kwargs...)

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
                    iterations = local_maxiters,
                )
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

function SciMLBase.__init(
        prob::OptimizationProblem,
        opt::Union{
            Optim.AbstractOptimizer, Optim.Fminbox,
            Optim.SAMIN, Optim.ConstrainedOptimizer,
        };
        callback = (args...) -> (false),
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        progress = false,
        kwargs...
    )
    if !isnothing(prob.lb) || !isnothing(prob.ub)
        if !(opt isa Union{Optim.Fminbox, Optim.SAMIN, Optim.AbstractConstrainedOptimizer})
            if opt isa Optim.ParticleSwarm
                opt = Optim.ParticleSwarm(;
                    lower = prob.lb, upper = prob.ub,
                    n_particles = opt.n_particles
                )
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

    maxiters = OptimizationBase._check_and_convert_maxiters(maxiters)
    maxtime = OptimizationBase._check_and_convert_maxtime(maxtime)
    return OptimizationCache(
        prob, opt; callback, maxiters, maxtime, abstol,
        reltol, progress,
        kwargs...
    )
end

function SciMLBase.__solve(cache::OptimizationCache{O}) where {O <: Optim.AbstractOptimizer}
    local x, cur, state
    !(cache.opt isa Optim.ZerothOrderOptimizer) && cache.f.grad === nothing &&
        error("Use OptimizationFunction to pass the derivatives or automatically generate them with one of the autodiff backends")

    iter_counter = Ref(0)
    function _cb(state_or_trace)
        iter_counter[] += 1
        cb_state = _extract_callback_state(state_or_trace, cache.opt, iter_counter)
        opt_state = OptimizationBase.OptimizationState(
            iter = cb_state.iter,
            u = cb_state.θ,
            p = cache.p,
            objective = cb_state.loss_val,
            grad = cb_state.g,
            hess = cb_state.h,
            original = cb_state.original
        )
        cb_call = cache.callback(opt_state, cb_state.loss_val)
        if !(cb_call isa Bool)
            error("The callback should return a boolean `halt` for whether to stop the optimization process.")
        end
        return cb_call
    end

    _loss = function (θ)
        x = cache.f.f(θ, cache.p)
        __x = first(x)
        return cache.sense === OptimizationBase.MaxSense ? -__x : __x
    end

    if cache.f.fg === nothing
        fg! = function (G, θ)
            if G !== nothing
                cache.f.grad(G, θ)
                if cache.sense === OptimizationBase.MaxSense
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
            return if cache.sense === OptimizationBase.MaxSense
                H .*= -one(eltype(H))
            end
        end
        optim_f = _make_hv_objective(_loss, fg!, hv, cache.u0)
    else
        gg = function (G, θ)
            cache.f.grad(G, θ)
            return if cache.sense === OptimizationBase.MaxSense
                G .*= -one(eltype(G))
            end
        end

        hh = function (H, θ)
            cache.f.hess(H, θ)
            return if cache.sense === OptimizationBase.MaxSense
                H .*= -one(eltype(H))
            end
        end
        u0_type = eltype(cache.u0)
        optim_f = Optim.TwiceDifferentiable(
            _loss, gg, fg!, hh, cache.u0,
            real(zero(u0_type)),
            Optim.NLSolversBase.alloc_DF(
                cache.u0,
                real(zero(u0_type))
            ),
            isnothing(cache.f.hess_prototype) ?
                Optim.NLSolversBase.alloc_H(
                    cache.u0,
                    real(zero(u0_type))
                ) :
                similar(cache.f.hess_prototype, u0_type)
        )
    end

    opt_args = __map_optimizer_args(
        cache, cache.opt, callback = _cb,
        maxiters = cache.solver_args.maxiters,
        maxtime = cache.solver_args.maxtime,
        abstol = cache.solver_args.abstol,
        reltol = cache.solver_args.reltol;
        cache.solver_args...
    )

    t0 = time()
    opt_res = Optim.optimize(optim_f, cache.u0, cache.opt, opt_args)
    t1 = time()
    opt_ret = Symbol(Optim.converged(opt_res))
    stats = OptimizationBase.OptimizationStats(;
        iterations = opt_res.iterations,
        time = t1 - t0, fevals = opt_res.f_calls, gevals = opt_res.g_calls,
        hevals = opt_res.h_calls
    )
    return SciMLBase.build_solution(
        cache, cache.opt,
        opt_res.minimizer,
        cache.sense === OptimizationBase.MaxSense ? -opt_res.minimum :
            opt_res.minimum; original = opt_res, retcode = opt_ret,
        stats = stats
    )
end

function SciMLBase.__solve(cache::OptimizationCache{O}) where {
        O <: Union{
            Optim.Fminbox, Optim.SAMIN,
        },
    }
    local x, cur, state

    iter_counter = Ref(0)
    # For Fminbox, the inner method determines NelderMead behavior
    inner_opt = cache.opt isa Optim.SAMIN ? cache.opt : cache.opt.method
    function _cb(state_or_trace)
        iter_counter[] += 1
        cb_state = _extract_callback_state(state_or_trace, inner_opt, iter_counter)
        opt_state = OptimizationBase.OptimizationState(
            iter = cb_state.iter,
            u = cb_state.θ,
            p = cache.p,
            objective = cb_state.loss_val,
            grad = cb_state.g,
            hess = cb_state.h,
            original = cb_state.original
        )
        cb_call = cache.callback(opt_state, cb_state.loss_val)
        if !(cb_call isa Bool)
            error("The callback should return a boolean `halt` for whether to stop the optimization process.")
        end
        return cb_call
    end

    _loss = function (θ)
        x = cache.f.f(θ, cache.p)
        __x = first(x)
        return cache.sense === OptimizationBase.MaxSense ? -__x : __x
    end

    if cache.f.fg === nothing
        fg! = function (G, θ)
            if G !== nothing
                cache.f.grad(G, θ)
                if cache.sense === OptimizationBase.MaxSense
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
        return if cache.sense === OptimizationBase.MaxSense
            G .*= -one(eltype(G))
        end
    end
    optim_f = Optim.OnceDifferentiable(_loss, gg, fg!, cache.u0)

    opt_args = __map_optimizer_args(
        cache, cache.opt, callback = _cb,
        maxiters = cache.solver_args.maxiters,
        maxtime = cache.solver_args.maxtime,
        abstol = cache.solver_args.abstol,
        reltol = cache.solver_args.reltol;
        cache.solver_args...
    )

    t0 = time()
    opt_res = Optim.optimize(optim_f, cache.lb, cache.ub, cache.u0, cache.opt, opt_args)
    t1 = time()
    opt_ret = Symbol(Optim.converged(opt_res))
    stats = OptimizationBase.OptimizationStats(;
        iterations = opt_res.iterations,
        time = t1 - t0, fevals = opt_res.f_calls, gevals = opt_res.g_calls,
        hevals = opt_res.h_calls
    )
    return SciMLBase.build_solution(
        cache, cache.opt,
        opt_res.minimizer, opt_res.minimum;
        original = opt_res, retcode = opt_ret, stats = stats
    )
end

function SciMLBase.__solve(cache::OptimizationCache{O}) where {
        O <:
        Optim.ConstrainedOptimizer,
    }
    local x, cur, state

    iter_counter = Ref(0)
    function _cb(state_or_trace)
        iter_counter[] += 1
        cb_state = _extract_callback_state(state_or_trace, cache.opt, iter_counter)
        opt_state = OptimizationBase.OptimizationState(
            iter = cb_state.iter,
            u = cb_state.θ,
            p = cache.p,
            objective = cb_state.loss_val,
            grad = cb_state.g,
            hess = cb_state.h,
            original = cb_state.original
        )
        cb_call = cache.callback(opt_state, cb_state.loss_val)
        if !(cb_call isa Bool)
            error("The callback should return a boolean `halt` for whether to stop the optimization process.")
        end
        return cb_call
    end

    _loss = function (θ)
        x = cache.f.f(θ, cache.p)
        __x = first(x)
        return cache.sense === OptimizationBase.MaxSense ? -__x : __x
    end

    if cache.f.fg === nothing
        fg! = function (G, θ)
            if G !== nothing
                cache.f.grad(G, θ)
                if cache.sense === OptimizationBase.MaxSense
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
        return if cache.sense === OptimizationBase.MaxSense
            G .*= -one(eltype(G))
        end
    end

    hh = function (H, θ)
        cache.f.hess(H, θ)
        return if cache.sense === OptimizationBase.MaxSense
            H .*= -one(eltype(H))
        end
    end
    u0_type = eltype(cache.u0)

    optim_f = if SciMLBase.requireshessian(cache.opt)
        Optim.TwiceDifferentiable(
            _loss, gg, fg!, hh, cache.u0,
            real(zero(u0_type)),
            Optim.NLSolversBase.alloc_DF(
                cache.u0,
                real(zero(u0_type))
            ),
            isnothing(cache.f.hess_prototype) ?
                Optim.NLSolversBase.alloc_H(
                    cache.u0,
                    real(zero(u0_type))
                ) :
                similar(cache.f.hess_prototype, u0_type)
        )
    else
        Optim.OnceDifferentiable(
            _loss, gg, fg!, cache.u0,
            real(zero(u0_type)),
            Optim.NLSolversBase.alloc_DF(
                cache.u0,
                real(zero(u0_type))
            )
        )
    end

    cons_hl! = function (h, θ, λ)
        res = [similar(h) for i in 1:length(λ)]
        cache.f.cons_h(res, θ)
        for i in 1:length(λ)
            h .+= λ[i] * res[i]
        end
        return
    end

    lb = cache.lb === nothing ? [] : cache.lb
    ub = cache.ub === nothing ? [] : cache.ub

    optim_fc = if SciMLBase.requireshessian(cache.opt)
        if cache.f.cons !== nothing
            Optim.TwiceDifferentiableConstraints(
                cache.f.cons, cache.f.cons_j,
                cons_hl!,
                lb, ub,
                cache.lcons, cache.ucons
            )
        else
            Optim.TwiceDifferentiableConstraints(lb, ub)
        end
    else
        if cache.f.cons !== nothing
            Optim.OnceDifferentiableConstraints(
                cache.f.cons, cache.f.cons_j,
                lb, ub,
                cache.lcons, cache.ucons
            )
        else
            Optim.OnceDifferentiableConstraints(lb, ub)
        end
    end

    opt_args = __map_optimizer_args(
        cache, cache.opt, callback = _cb,
        maxiters = cache.solver_args.maxiters,
        maxtime = cache.solver_args.maxtime,
        abstol = cache.solver_args.abstol,
        reltol = cache.solver_args.reltol;
        cache.solver_args...
    )

    t0 = time()
    if lb === nothing && ub === nothing && cache.f.cons === nothing
        opt_res = Optim.optimize(optim_f, cache.u0, cache.opt, opt_args)
    else
        opt_res = Optim.optimize(optim_f, optim_fc, cache.u0, cache.opt, opt_args)
    end
    t1 = time()
    opt_ret = Symbol(Optim.converged(opt_res))
    stats = OptimizationBase.OptimizationStats(;
        iterations = opt_res.iterations,
        time = t1 - t0, fevals = opt_res.f_calls, gevals = opt_res.g_calls,
        hevals = opt_res.h_calls
    )
    return SciMLBase.build_solution(
        cache, cache.opt,
        opt_res.minimizer, opt_res.minimum;
        original = opt_res, retcode = opt_ret,
        stats = stats
    )
end

end
