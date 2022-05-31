module OptimizationOptimJL

using Reexport, Optimization, Optimization.SciMLBase
@reexport using Optim
decompose_trace(trace::Optim.OptimizationTrace) = last(trace)
decompose_trace(trace::Optim.OptimizationState) = trace

function __map_optimizer_args(prob::OptimizationProblem, opt::Union{Optim.AbstractOptimizer,Optim.Fminbox,Optim.SAMIN,Optim.ConstrainedOptimizer};
    callback=nothing,
    maxiters::Union{Number,Nothing}=nothing,
    maxtime::Union{Number,Nothing}=nothing,
    abstol::Union{Number,Nothing}=nothing,
    reltol::Union{Number,Nothing}=nothing,
    kwargs...)
    if !isnothing(abstol)
        @warn "common abstol is currently not used by $(opt)"
    end

    mapped_args = (; extended_trace=true, kwargs...)

    if !isnothing(callback)
        mapped_args = (; mapped_args..., callback=callback)
    end

    if !isnothing(maxiters)
        mapped_args = (; mapped_args..., iterations=maxiters)
    end

    if !isnothing(maxtime)
        mapped_args = (; mapped_args..., time_limit=maxtime)
    end

    if !isnothing(reltol)
        mapped_args = (; mapped_args..., f_tol=reltol)
    end

    return Optim.Options(; mapped_args...)
end

function SciMLBase.__solve(prob::OptimizationProblem,
    opt::Optim.AbstractOptimizer,
    data=Optimization.DEFAULT_DATA;
    kwargs...)
    if !isnothing(prob.lb) | !isnothing(prob.ub)
        if !(opt isa Union{Optim.Fminbox,Optim.SAMIN,Optim.AbstractConstrainedOptimizer})
            if opt isa Optim.ParticleSwarm
                opt = Optim.ParticleSwarm(; lower=prob.lb, upper=prob.ub, n_particles=opt.n_particles)
            elseif opt isa Optim.SimulatedAnnealing
                @warn "$(opt) can currently not be wrapped in Fminbox(). The lower and upper bounds thus will be ignored. Consider using a different optimizer or open an issue with Optim.jl"
            else
                opt = Optim.Fminbox(opt)
            end
        end
    end

    return ___solve(prob, opt, data; kwargs...)
end

function ___solve(prob::OptimizationProblem, opt::Optim.AbstractOptimizer,
    data=Optimization.DEFAULT_DATA;
    callback=(args...) -> (false),
    maxiters::Union{Number,Nothing}=nothing,
    maxtime::Union{Number,Nothing}=nothing,
    abstol::Union{Number,Nothing}=nothing,
    reltol::Union{Number,Nothing}=nothing,
    progress=false,
    kwargs...)

    local x, cur, state

    if data != Optimization.DEFAULT_DATA
        maxiters = length(data)
    end

    cur, state = iterate(data)

    function _cb(trace)
        cb_call = opt isa Optim.NelderMead ? callback(decompose_trace(trace).metadata["centroid"], x...) : callback(decompose_trace(trace).metadata["x"], x...)
        if !(typeof(cb_call) <: Bool)
            error("The callback should return a boolean `halt` for whether to stop the optimization process.")
        end
        nx_itr = iterate(data, state)
        if isnothing(nx_itr)
            true
        else
            cur, state = nx_itr
            cb_call
        end
    end

    maxiters = Optimization._check_and_convert_maxiters(maxiters)
    maxtime = Optimization._check_and_convert_maxtime(maxtime)

    f = Optimization.instantiate_function(prob.f, prob.u0, prob.f.adtype, prob.p)

    !(opt isa Optim.ZerothOrderOptimizer) && f.grad === nothing && error("Use OptimizationFunction to pass the derivatives or automatically generate them with one of the autodiff backends")

    _loss = function (θ)
        x = f.f(θ, prob.p, cur...)
        __x = first(x)
        return prob.sense === Optimization.MaxSense ? -__x : __x
    end

    fg! = function (G, θ)
        if G !== nothing
            f.grad(G, θ, cur...)
            if prob.sense === Optimization.MaxSense
                G .*= false
            end
        end
        return _loss(θ)
    end

    if opt isa Optim.KrylovTrustRegion
        hv = function (H, θ, v)
            f.hv(H, θ, v, cur...)
            if prob.sense === Optimization.MaxSense
                H .*= false
            end
        end
        optim_f = Optim.TwiceDifferentiableHV(_loss, fg!, hv, prob.u0)
    else
        gg = function (G, θ)
            f.grad(G, θ, cur...)
            if prob.sense === Optimization.MaxSense
                G .*= false
            end
        end

        hh = function (H, θ)
            f.hess(H, θ, cur...)
            if prob.sense === Optimization.MaxSense
                H .*= false
            end
        end
        optim_f = Optim.TwiceDifferentiable(_loss, gg, fg!, hh, prob.u0)
    end

    opt_args = __map_optimizer_args(prob, opt, callback=_cb, maxiters=maxiters, maxtime=maxtime, abstol=abstol, reltol=reltol; kwargs...)

    t0 = time()
    opt_res = Optim.optimize(optim_f, prob.u0, opt, opt_args)
    t1 = time()
    opt_ret = Symbol(Optim.converged(opt_res))

    SciMLBase.build_solution(prob, opt, opt_res.minimizer, prob.sense === Optimization.MaxSense ? -opt_res.minimum : opt_res.minimum; original=opt_res, retcode=opt_ret)
end

function ___solve(prob::OptimizationProblem, opt::Union{Optim.Fminbox,Optim.SAMIN},
    data=Optimization.DEFAULT_DATA;
    callback=(args...) -> (false),
    maxiters::Union{Number,Nothing}=nothing,
    maxtime::Union{Number,Nothing}=nothing,
    abstol::Union{Number,Nothing}=nothing,
    reltol::Union{Number,Nothing}=nothing,
    progress=false,
    kwargs...)

    local x, cur, state

    if data != Optimization.DEFAULT_DATA
        maxiters = length(data)
    end

    cur, state = iterate(data)

    function _cb(trace)
        cb_call = !(opt isa Optim.SAMIN) && opt.method == Optim.NelderMead() ? callback(decompose_trace(trace).metadata["centroid"], x...) : callback(decompose_trace(trace).metadata["x"], x...)
        if !(typeof(cb_call) <: Bool)
            error("The callback should return a boolean `halt` for whether to stop the optimization process.")
        end
        nx_itr = iterate(data, state)
        if isnothing(nx_itr)
            true
        else
            cur, state = nx_itr
            cb_call
        end
    end

    maxiters = Optimization._check_and_convert_maxiters(maxiters)
    maxtime = Optimization._check_and_convert_maxtime(maxtime)

    f = Optimization.instantiate_function(prob.f, prob.u0, prob.f.adtype, prob.p)

    !(opt isa Optim.ZerothOrderOptimizer) && f.grad === nothing && error("Use OptimizationFunction to pass the derivatives or automatically generate them with one of the autodiff backends")

    _loss = function (θ)
        x = f.f(θ, prob.p, cur...)
        __x = first(x)
        return prob.sense === Optimization.MaxSense ? -__x : __x
    end
    fg! = function (G, θ)
        if G !== nothing
            f.grad(G, θ, cur...)
            if prob.sense === Optimization.MaxSense
                G .*= false
            end
        end
        return _loss(θ)
    end

    gg = function (G, θ)
        f.grad(G, θ, cur...)
        if prob.sense === Optimization.MaxSense
            G .*= false
        end
    end
    optim_f = Optim.OnceDifferentiable(_loss, gg, fg!, prob.u0)

    opt_args = __map_optimizer_args(prob, opt, callback=_cb, maxiters=maxiters, maxtime=maxtime, abstol=abstol, reltol=reltol; kwargs...)

    t0 = time()
    opt_res = Optim.optimize(optim_f, prob.lb, prob.ub, prob.u0, opt, opt_args)
    t1 = time()
    opt_ret = Symbol(Optim.converged(opt_res))

    SciMLBase.build_solution(prob, opt, opt_res.minimizer, opt_res.minimum; original=opt_res, retcode=opt_ret)
end


function ___solve(prob::OptimizationProblem, opt::Optim.ConstrainedOptimizer,
    data=Optimization.DEFAULT_DATA;
    callback=(args...) -> (false),
    maxiters::Union{Number,Nothing}=nothing,
    maxtime::Union{Number,Nothing}=nothing,
    abstol::Union{Number,Nothing}=nothing,
    reltol::Union{Number,Nothing}=nothing,
    progress=false,
    kwargs...)

    local x, cur, state

    if data != Optimization.DEFAULT_DATA
        maxiters = length(data)
    end

    cur, state = iterate(data)

    function _cb(trace)
        cb_call = callback(decompose_trace(trace).metadata["x"], x...)
        if !(typeof(cb_call) <: Bool)
            error("The callback should return a boolean `halt` for whether to stop the optimization process.")
        end
        nx_itr = iterate(data, state)
        if isnothing(nx_itr)
            true
        else
            cur, state = nx_itr
            cb_call
        end
    end

    maxiters = Optimization._check_and_convert_maxiters(maxiters)
    maxtime = Optimization._check_and_convert_maxtime(maxtime)

    f = Optimization.instantiate_function(prob.f, prob.u0, prob.f.adtype, prob.p, prob.ucons === nothing ? 0 : length(prob.ucons))

    f.cons_j === nothing && error("This optimizer requires derivative definitions for nonlinear constraints. If the problem does not have nonlinear constraints, choose a different optimizer. Otherwise define the derivative for cons using OptimizationFunction either directly or automatically generate them with one of the autodiff backends")

    _loss = function (θ)
        x = f.f(θ, prob.p, cur...)
        __x = first(x)
        return prob.sense === Optimization.MaxSense ? -__x : __x
    end
    fg! = function (G, θ)
        if G !== nothing
            f.grad(G, θ, cur...)
            if prob.sense === Optimization.MaxSense
                G .*= false
            end
        end
        return _loss(θ)
    end
    gg = function (G, θ)
        f.grad(G, θ, cur...)
        if prob.sense === Optimization.MaxSense
            G .*= false
        end
    end

    hh = function (H, θ)
        f.hess(H, θ, cur...)
        if prob.sense === Optimization.MaxSense
            H .*= false
        end
    end
    optim_f = Optim.TwiceDifferentiable(_loss, gg, fg!, hh, prob.u0)

    cons! = (res, θ) -> res .= f.cons(θ)

    cons_j! = function (J, x)
        f.cons_j(J, x)
    end

    cons_hl! = function (h, θ, λ)
        res = [similar(h) for i in 1:length(λ)]
        f.cons_h(res, θ)
        for i in 1:length(λ)
            h .+= λ[i] * res[i]
        end
    end

    lb = prob.lb === nothing ? [] : prob.lb
    ub = prob.ub === nothing ? [] : prob.ub
    optim_fc = Optim.TwiceDifferentiableConstraints(cons!, cons_j!, cons_hl!, lb, ub, prob.lcons, prob.ucons)

    opt_args = __map_optimizer_args(prob, opt, callback=_cb, maxiters=maxiters, maxtime=maxtime, abstol=abstol, reltol=reltol; kwargs...)

    t0 = time()
    opt_res = Optim.optimize(optim_f, optim_fc, prob.u0, opt, opt_args)
    t1 = time()
    opt_ret = Symbol(Optim.converged(opt_res))

    SciMLBase.build_solution(prob, opt, opt_res.minimizer, opt_res.minimum; original=opt_res, retcode=opt_ret)
end


end
