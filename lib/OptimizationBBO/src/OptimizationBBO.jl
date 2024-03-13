module OptimizationBBO

using Reexport
import Optimization
import BlackBoxOptim, Optimization.SciMLBase

abstract type BBO end

SciMLBase.requiresbounds(::BBO) = true
SciMLBase.allowsbounds(::BBO) = true
SciMLBase.supports_opt_cache_interface(opt::BBO) = true





for j in string.(BlackBoxOptim.SingleObjectiveMethodNames)
    eval(Meta.parse("Base.@kwdef struct BBO_" * j * " <: BBO method=:" * j * " end"))
    eval(Meta.parse("export BBO_" * j))
end

function decompose_trace(opt::BlackBoxOptim.OptRunController, progress)
    if progress
        maxiters = opt.max_steps
        max_time = opt.max_time
        msg = "loss: " *
              sprint(show, BlackBoxOptim.best_fitness(opt), context = :compact => true)
        if iszero(max_time)
            # we stop at either convergence or max_steps
            n_steps = BlackBoxOptim.num_steps(opt)
            Base.@logmsg(Base.LogLevel(-1), msg, progress=n_steps / maxiters,
                _id=:OptimizationBBO)
        else
            # we stop at either convergence or max_time
            elapsed = BlackBoxOptim.elapsed_time(opt)
            Base.@logmsg(Base.LogLevel(-1), msg, progress=elapsed / max_time,
                _id=:OptimizationBBO)
        end
    end
    return BlackBoxOptim.best_candidate(opt)
end

function __map_optimizer_args(prob::Optimization.OptimizationCache, opt::BBO;
        callback = nothing,
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        verbose::Bool = false,
        kwargs...)
    if !isnothing(reltol)
        @warn "common reltol is currently not used by $(opt)"
    end
    mapped_args = (; kwargs...)
    mapped_args = (; mapped_args..., Method = opt.method,
        SearchRange = [(prob.lb[i], prob.ub[i]) for i in 1:length(prob.lb)])

    if !isnothing(callback)
        mapped_args = (; mapped_args..., CallbackFunction = callback,
            CallbackInterval = 0.0)
    end

    if !isnothing(maxiters)
        mapped_args = (; mapped_args..., MaxSteps = maxiters)
    end

    if !isnothing(maxtime)
        mapped_args = (; mapped_args..., MaxTime = maxtime)
    end

    if !isnothing(abstol)
        mapped_args = (; mapped_args..., MinDeltaFitnessTolerance = abstol)
    end

    if verbose
        mapped_args = (; mapped_args..., TraceMode = :verbose)
    else
        mapped_args = (; mapped_args..., TraceMode = :silent)
    end

    return mapped_args
end

function SciMLBase.__solve(cache::Optimization.OptimizationCache{
        F,
        RC,
        LB,
        UB,
        LC,
        UC,
        S,
        O,
        D,
        P,
        C
}) where {
        F,
        RC,
        LB,
        UB,
        LC,
        UC,
        S,
        O <:
        BBO,
        D,
        P,
        C
}
    local x, cur, state

    if cache.data != Optimization.DEFAULT_DATA
        maxiters = length(cache.data)
    end

    cur, state = iterate(cache.data)

    function _cb(trace)
        if cache.callback === Optimization.DEFAULT_CALLBACK
            cb_call = false
        else
            n_steps = BlackBoxOptim.num_steps(trace)
            curr_u = decompose_trace(trace, cache.progress)
            opt_state = Optimization.OptimizationState(;
                iter = n_steps,
                u = curr_u,
                objective = x[1],
                original = trace)
            cb_call = cache.callback(opt_state, x...)
        end

        if !(cb_call isa Bool)
            error("The callback should return a boolean `halt` for whether to stop the optimization process.")
        end
        if cb_call == true
            BlackBoxOptim.shutdown_optimizer!(trace) #doesn't work
        end

        if cache.data !== Optimization.DEFAULT_DATA
            cur, state = iterate(cache.data, state)
        end
        cb_call
    end

    maxiters = Optimization._check_and_convert_maxiters(cache.solver_args.maxiters)
    maxtime = Optimization._check_and_convert_maxtime(cache.solver_args.maxtime)

    _loss = function (θ)
        if cache.callback === Optimization.DEFAULT_CALLBACK &&
           cache.data === Optimization.DEFAULT_DATA
            return first(cache.f(θ, cache.p))
        elseif cache.callback === Optimization.DEFAULT_CALLBACK
            return first(cache.f(θ, cache.p, cur...))
        elseif cache.data !== Optimization.DEFAULT_DATA
            x = cache.f(θ, cache.p)
            return first(x)
        else
            x = cache.f(θ, cache.p, cur...)
            return first(x)
        end
    end

    opt_args = __map_optimizer_args(cache, cache.opt;
        callback = cache.callback === Optimization.DEFAULT_CALLBACK &&
                   cache.data === Optimization.DEFAULT_DATA ?
                   nothing : _cb,
        cache.solver_args...,
        maxiters = maxiters,
        maxtime = maxtime)

    opt_setup = BlackBoxOptim.bbsetup(_loss; opt_args...)

    t0 = time()

    if isnothing(cache.u0)
        opt_res = BlackBoxOptim.bboptimize(opt_setup)
    else
        opt_res = BlackBoxOptim.bboptimize(opt_setup, cache.u0)
    end

    if cache.progress
        # Set progressbar to 1 to finish it
        Base.@logmsg(Base.LogLevel(-1), "", progress=1, _id=:OptimizationBBO)
    end

    t1 = time()

    opt_ret = Symbol(opt_res.stop_reason)
    stats = Optimization.OptimizationStats(;
        iterations = opt_res.iterations,
        time = t1 - t0,
        fevals = opt_res.f_calls)
    SciMLBase.build_solution(cache, cache.opt,
        BlackBoxOptim.best_candidate(opt_res),
        BlackBoxOptim.best_fitness(opt_res);
        original = opt_res,
        retcode = opt_ret,
        stats = stats)
end

end
