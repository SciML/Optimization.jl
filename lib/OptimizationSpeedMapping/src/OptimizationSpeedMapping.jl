module OptimizationSpeedMapping

using Reexport
@reexport using OptimizationBase
using SpeedMapping, SciMLBase

export SpeedMappingOpt

struct SpeedMappingOpt end

SciMLBase.allowsbounds(::SpeedMappingOpt) = true
SciMLBase.allowscallback(::SpeedMappingOpt) = false
SciMLBase.has_init(opt::SpeedMappingOpt) = true
SciMLBase.requiresgradient(opt::SpeedMappingOpt) = true

function __map_optimizer_args(
        cache::OptimizationBase.OptimizationCache, opt::SpeedMappingOpt;
        callback = nothing,
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing
    )

    # add optimiser options from kwargs
    mapped_args = (;)

    if !(isnothing(maxiters))
        @info "maxiters defines maximum gradient calls for $(opt)"
        mapped_args = (; mapped_args..., maps_limit = maxiters)
    end

    if !(isnothing(maxtime))
        mapped_args = (; mapped_args..., time_limit = maxtime)
    end

    if !isnothing(abstol)
        @SciMLMessage(lazy"common abstol is currently not used by $(opt)",
            cache.verbose, :unsupported_kwargs)
    end

    if !isnothing(reltol)
        @SciMLMessage(lazy"common reltol is currently not used by $(opt)",
            cache.verbose, :unsupported_kwargs)
    end

    return mapped_args
end

function SciMLBase.__solve(cache::OptimizationCache{O}) where {O <: SpeedMappingOpt}
    local x

    _loss = function (θ)
        x = cache.f.f(θ, cache.p)
        return first(x)
    end

    if isnothing(cache.f.grad)
        @info "SpeedMapping's ForwardDiff AD backend is used to calculate the gradient information."
    end

    maxiters = OptimizationBase._check_and_convert_maxiters(cache.solver_args.maxiters)
    maxtime = OptimizationBase._check_and_convert_maxtime(cache.solver_args.maxtime)
    opt_args = __map_optimizer_args(
        cache, cache.opt, maxiters = maxiters,
        maxtime = maxtime,
        abstol = cache.solver_args.abstol,
        reltol = cache.solver_args.reltol; cache.solver_args...
    )

    t0 = time()
    opt_res = SpeedMapping.speedmapping(
        cache.u0; f = _loss, (g!) = cache.f.grad,
        lower = cache.lb,
        upper = cache.ub, opt_args...
    )
    t1 = time()
    opt_ret = Symbol(opt_res.converged)
    stats = OptimizationBase.OptimizationStats(; time = t1 - t0)
    return SciMLBase.build_solution(
        cache, cache.opt,
        opt_res.minimizer, _loss(opt_res.minimizer);
        original = opt_res, retcode = opt_ret, stats = stats
    )
end

end
