module OptimizationCMAEvolutionStrategy

using Reexport
@reexport using OptimizationBase
using CMAEvolutionStrategy
using OptimizationBase: SciMLBase

export CMAEvolutionStrategyOpt

struct CMAEvolutionStrategyOpt end

SciMLBase.allowsbounds(::CMAEvolutionStrategyOpt) = true
OptimizationBase.supports_opt_cache_interface(opt::CMAEvolutionStrategyOpt) = true
SciMLBase.requiresgradient(::CMAEvolutionStrategyOpt) = false
SciMLBase.requireshessian(::CMAEvolutionStrategyOpt) = false
SciMLBase.requiresconsjac(::CMAEvolutionStrategyOpt) = false
SciMLBase.requiresconshess(::CMAEvolutionStrategyOpt) = false

function __map_optimizer_args(prob::OptimizationBase.OptimizationCache, opt::CMAEvolutionStrategyOpt;
        callback = nothing,
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        verbose::Bool = false)
    if !isnothing(reltol)
        @warn "common reltol is currently not used by $(opt)"
    end

    mapped_args = (; lower = prob.lb,
        upper = prob.ub,
        logger = CMAEvolutionStrategy.BasicLogger(prob.u0;
            verbosity = verbose ? 1 : 0,
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

function SciMLBase.__solve(cache::OptimizationBase.OptimizationCache{
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

    function _cb(opt, y, fvals, perm)
        curr_u = xbest(opt)
        opt_state = OptimizationBase.OptimizationState(; iter = length(opt.logger.fmedian),
            u = curr_u,
            p = cache.p,
            objective = fbest(opt),
            original = opt.logger)

        cb_call = cache.callback(opt_state, x...)
        if !(cb_call isa Bool)
            error("The callback should return a boolean `halt` for whether to stop the optimization process.")
        end
        cb_call
    end

    maxiters = OptimizationBase._check_and_convert_maxiters(cache.solver_args.maxiters)
    maxtime = OptimizationBase._check_and_convert_maxtime(cache.solver_args.maxtime)

    _loss = function (θ)
        x = cache.f(θ, cache.p)
        return first(x)
    end

    opt_args = __map_optimizer_args(cache, cache.opt; callback = _cb, cache.solver_args...,
        maxiters = maxiters,
        maxtime = maxtime)

    t0 = time()
    opt_res = CMAEvolutionStrategy.minimize(_loss, cache.u0, 0.1; opt_args...)
    t1 = time()

    opt_ret = opt_res.stop.reason
    stats = OptimizationBase.OptimizationStats(;
        iterations = length(opt_res.logger.fmedian),
        time = t1 - t0,
        fevals = length(opt_res.logger.fmedian))
    SciMLBase.build_solution(cache, cache.opt,
        xbest(opt_res),
        fbest(opt_res); original = opt_res,
        retcode = opt_ret,
        stats = stats)
end

end
