module OptimizationCMAEvolutionStrategy

using Reexport
@reexport using Optimization
using CMAEvolutionStrategy, Optimization.SciMLBase

export CMAEvolutionStrategyOpt

struct CMAEvolutionStrategyOpt end

SciMLBase.allowsbounds(::CMAEvolutionStrategyOpt) = true
SciMLBase.allowscallback(::CMAEvolutionStrategyOpt) = false #looks like `logger` kwarg can be used to pass it, so should be implemented
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
        upper = prob.ub)

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
    C,
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
    C,
}
    local x, cur, state

    if cache.data != Optimization.DEFAULT_DATA
        maxiters = length(cache.data)
    end

    cur, state = iterate(cache.data)

    function _cb(trace)
        cb_call = cache.callback(decompose_trace(trace).metadata["x"], trace.value...)
        if !(cb_call isa Bool)
            error("The callback should return a boolean `halt` for whether to stop the optimization process.")
        end
        cur, state = iterate(data, state)
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

    SciMLBase.build_solution(cache, cache.opt,
        opt_res.logger.xbest[end],
        opt_res.logger.fbest[end]; original = opt_res,
        retcode = opt_ret, solve_time = t1 - t0)
end

end
