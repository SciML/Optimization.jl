module OptimizationCMAEvolutionStrategy

using Reexport
@reexport using Optimization
using CMAEvolutionStrategy, Optimization.SciMLBase

export CMAEvolutionStrategyOpt

struct CMAEvolutionStrategyOpt end

SciMLBase.allowsbounds(::CMAEvolutionStrategyOpt) = true
SciMLBase.supports_opt_cache_interface(opt::CMAEvolutionStrategyOpt) = true

function __map_optimizer_args(prob::OptimizationCache, opt::CMAEvolutionStrategyOpt;
        callback = nothing,
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing)
    if !isnothing(reltol)
        @warn "common reltol is currently not used by $(opt)"
    end

    mapped_args = (; lower = prob.lb,
        upper = prob.ub,
        logger = CMAEvolutionStrategy.BasicLogger(prob.u0;
            verbosity = 0,
            callback = callback))

    if !isnothing(maxiters)
        mapped_args = (; mapped_args..., maxiter = maxiters)
    end

    if !isnothing(maxtime)
        mapped_args = (; mapped_args..., maxtime = maxtime)
    end

    if !isnothing(abstol)
        mapped_args = (; mapped_args..., ftol = abstol)
    end

    return mapped_args
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
        CMAEvolutionStrategyOpt,
        D,
        P,
        C
}
    local x, cur, state

    if cache.data != Optimization.DEFAULT_DATA
        maxiters = length(cache.data)
    end

    cur, state = iterate(cache.data)

    function _cb(opt, y, fvals, perm)
        curr_u = opt.logger.xbest[end]
        opt_state = Optimization.OptimizationState(; iter = length(opt.logger.fmedian),
            u = curr_u,
            objective = opt.logger.fbest[end],
            original = opt.logger)

        cb_call = cache.callback(opt_state, x...)
        if !(cb_call isa Bool)
            error("The callback should return a boolean `halt` for whether to stop the optimization process.")
        end
        cur, state = iterate(cache.data, state)
        cb_call
    end

    maxiters = Optimization._check_and_convert_maxiters(cache.solver_args.maxiters)
    maxtime = Optimization._check_and_convert_maxtime(cache.solver_args.maxtime)

    _loss = function (θ)
        x = cache.f(θ, cache.p, cur...)
        return first(x)
    end

    opt_args = __map_optimizer_args(cache, cache.opt; callback = _cb, cache.solver_args...,
        maxiters = maxiters,
        maxtime = maxtime)

    t0 = time()
    opt_res = CMAEvolutionStrategy.minimize(_loss, cache.u0, 0.1; opt_args...)
    t1 = time()

    opt_ret = opt_res.stop.reason
    stats = Optimization.OptimizationStats(;
        iterations = length(opt_res.logger.fmedian),
        time = t1 - t0,
        fevals = length(opt_res.logger.fmedian))
    SciMLBase.build_solution(cache, cache.opt,
        opt_res.logger.xbest[end],
        opt_res.logger.fbest[end]; original = opt_res,
        retcode = opt_ret,
        stats = stats)
end

end
