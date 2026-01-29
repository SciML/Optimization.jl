module OptimizationBBO

using Reexport
@reexport using OptimizationBase
using SciMLBase
using BlackBoxOptim: BlackBoxOptim

abstract type BBO end

SciMLBase.requiresbounds(::BBO) = true
SciMLBase.allowsbounds(::BBO) = true
SciMLBase.allowscallback(opt::BBO) = true
SciMLBase.has_init(opt::BBO) = true

for j in string.(BlackBoxOptim.SingleObjectiveMethodNames)
    eval(Meta.parse("Base.@kwdef struct BBO_" * j * " <: BBO method=:" * j * " end"))
    eval(Meta.parse("export BBO_" * j))
end

Base.@kwdef struct BBO_borg_moea <: BBO
    method = :borg_moea
end
export BBO_borg_moea

function decompose_trace(opt::BlackBoxOptim.OptRunController, progress)
    if progress
        maxiters = opt.max_steps
        max_time = opt.max_time
        msg = "loss: " *
            sprint(show, BlackBoxOptim.best_fitness(opt), context = :compact => true)
        if iszero(max_time)
            # we stop at either convergence or max_steps
            n_steps = BlackBoxOptim.num_steps(opt)
            Base.@logmsg(
                Base.LogLevel(-1), msg, progress = n_steps / maxiters,
                _id = :OptimizationBBO
            )
        else
            # we stop at either convergence or max_time
            elapsed = BlackBoxOptim.elapsed_time(opt)
            Base.@logmsg(
                Base.LogLevel(-1), msg, progress = elapsed / max_time,
                _id = :OptimizationBBO
            )
        end
    end
    return BlackBoxOptim.best_candidate(opt)
end

function __map_optimizer_args(
        prob::OptimizationCache, opt::BBO;
        callback = nothing,
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        verbose::Bool = false,
        kwargs...
    )
    if !isnothing(reltol)
        @SciMLMessage(
            lazy"common reltol is currently not used by $(opt)",
            prob.verbose, :unsupported_kwargs
        )
    end
    mapped_args = (; kwargs...)
    mapped_args = (;
        mapped_args..., Method = opt.method,
        SearchRange = [(prob.lb[i], prob.ub[i]) for i in 1:length(prob.lb)],
    )

    if !isnothing(callback)
        mapped_args = (;
            mapped_args..., CallbackFunction = callback,
            CallbackInterval = 0.0,
        )
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

# single objective
map_objective(obj) = obj
# multiobjective
function map_objective(obj::BlackBoxOptim.IndexedTupleFitness)
    return obj.orig
end

function SciMLBase.__solve(cache::OptimizationCache{O}) where {O <: BBO}
    function _cb(trace)
        if cache.callback === OptimizationBase.DEFAULT_CALLBACK
            cb_call = false
        else
            n_steps = BlackBoxOptim.num_steps(trace)
            curr_u = decompose_trace(trace, cache.progress)
            objective = map_objective(BlackBoxOptim.best_fitness(trace))
            opt_state = OptimizationBase.OptimizationState(;
                iter = n_steps,
                u = curr_u,
                p = cache.p,
                objective,
                original = trace
            )
            cb_call = cache.callback(opt_state, objective)
        end

        if !(cb_call isa Bool)
            error("The callback should return a boolean `halt` for whether to stop the optimization process.")
        end
        if cb_call == true
            BlackBoxOptim.shutdown_optimizer!(trace) #doesn't work
        end

        return cb_call
    end

    maxiters = OptimizationBase._check_and_convert_maxiters(cache.solver_args.maxiters)
    maxtime = OptimizationBase._check_and_convert_maxtime(cache.solver_args.maxtime)

    _loss = function (θ)
        return cache.f(θ, cache.p)
    end

    opt_args = __map_optimizer_args(
        cache, cache.opt;
        callback = cache.callback === OptimizationBase.DEFAULT_CALLBACK ?
            nothing : _cb,
        cache.solver_args...,
        maxiters = maxiters,
        maxtime = maxtime
    )

    opt_setup = BlackBoxOptim.bbsetup(_loss; opt_args...)

    if isnothing(cache.u0)
        opt_res = BlackBoxOptim.bboptimize(opt_setup)
    else
        opt_res = BlackBoxOptim.bboptimize(opt_setup, cache.u0)
    end

    if cache.progress
        # Set progressbar to 1 to finish it
        Base.@logmsg(Base.LogLevel(-1), "", progress = 1, _id = :OptimizationBBO)
    end

    # Use the improved convert function
    opt_ret = OptimizationBase.deduce_retcode(opt_res.stop_reason, cache.verbose)
    stats = OptimizationBase.OptimizationStats(;
        iterations = opt_res.iterations,
        time = opt_res.elapsed_time,
        fevals = opt_res.f_calls
    )
    return SciMLBase.build_solution(
        cache, cache.opt,
        BlackBoxOptim.best_candidate(opt_res),
        BlackBoxOptim.best_fitness(opt_res);
        original = opt_res,
        retcode = opt_ret,
        stats = stats
    )
end

end
