module OptimizationSpeedMapping

using Reexport
@reexport using Optimization
using SpeedMapping, Optimization.SciMLBase

export SpeedMappingOpt

struct SpeedMappingOpt end

SciMLBase.allowsbounds(::SpeedMappingOpt) = true
SciMLBase.allowscallback(::SpeedMappingOpt) = false

struct SpeedMappingOptimizationCache{F <: OptimizationFunction, RC, LB, UB, O, P} <:
       SciMLBase.AbstractOptimizationCache
    f::F
    reinit_cache::RC
    lb::LB
    ub::UB
    opt::O
    progress::P
    solver_args::NamedTuple
end

function Base.getproperty(cache::SpeedMappingOptimizationCache, x::Symbol)
    if x in fieldnames(Optimization.ReInitCache)
        return getfield(cache.reinit_cache, x)
    end
    return getfield(cache, x)
end

function SpeedMappingOptimizationCache(prob::OptimizationProblem, opt; progress, kwargs...)
    reinit_cache = Optimization.ReInitCache(prob.u0, prob.p) # everything that can be changed via `reinit`
    f = Optimization.instantiate_function(prob.f, reinit_cache, prob.f.adtype)
    return SpeedMappingOptimizationCache(f, reinit_cache, prob.lb, prob.ub, opt, progress,
                                         NamedTuple(kwargs))
end

SciMLBase.supports_opt_cache_interface(opt::SpeedMappingOpt) = true
SciMLBase.has_reinit(cache::SpeedMappingOptimizationCache) = true
function SciMLBase.reinit!(cache::SpeedMappingOptimizationCache; p = missing, u0 = missing)
    if p === missing && u0 === missing
        p, u0 = cache.p, cache.u0
    else # at least one of them has a value
        if p === missing
            p = cache.p
        end
        if u0 === missing
            u0 = cache.u0
        end
        if (eltype(p) <: Pair && !isempty(p)) || (eltype(u0) <: Pair && !isempty(u0)) # one is a non-empty symbolic map
            hasproperty(cache.f, :sys) && hasfield(typeof(cache.f.sys), :ps) ||
                throw(ArgumentError("This cache does not support symbolic maps with `remake`, i.e. it does not have a symbolic origin." *
                                    " Please use `remake` with the `p` keyword argument as a vector of values, paying attention to parameter order."))
            hasproperty(cache.f, :sys) && hasfield(typeof(cache.f.sys), :states) ||
                throw(ArgumentError("This cache does not support symbolic maps with `remake`, i.e. it does not have a symbolic origin." *
                                    " Please use `remake` with the `u0` keyword argument as a vector of values, paying attention to state order."))
            p, u0 = SciMLBase.process_p_u0_symbolic(cache, p, u0)
        end
    end

    cache.reinit_cache.p = p
    cache.reinit_cache.u0 = u0

    return cache
end

function __map_optimizer_args(cache::SpeedMappingOptimizationCache, opt::SpeedMappingOpt;
                              callback = nothing,
                              maxiters::Union{Number, Nothing} = nothing,
                              maxtime::Union{Number, Nothing} = nothing,
                              abstol::Union{Number, Nothing} = nothing,
                              reltol::Union{Number, Nothing} = nothing)

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
        @warn "common abstol is currently not used by $(opt)"
    end

    if !isnothing(reltol)
        @warn "common reltol is currently not used by $(opt)"
    end

    return mapped_args
end

function SciMLBase.__init(prob::OptimizationProblem, opt::SpeedMappingOpt;
                          maxiters::Union{Number, Nothing} = nothing,
                          maxtime::Union{Number, Nothing} = nothing,
                          abstol::Union{Number, Nothing} = nothing,
                          reltol::Union{Number, Nothing} = nothing,
                          progress = false,
                          kwargs...)
    return SpeedMappingOptimizationCache(prob, opt; maxiters, maxtime, abstol, reltol,
                                         progress, kwargs...)
end

function SciMLBase.__solve(cache::SpeedMappingOptimizationCache)
    local x

    _loss = function (θ)
        x = cache.f.f(θ, cache.p)
        return first(x)
    end

    if isnothing(cache.f.grad)
        @info "SpeedMapping's ForwardDiff AD backend is used to calculate the gradient information."
    end

    maxiters = Optimization._check_and_convert_maxiters(cache.solver_args.maxiters)
    maxtime = Optimization._check_and_convert_maxtime(cache.solver_args.maxtime)
    opt_args = __map_optimizer_args(cache, cache.opt, maxiters = maxiters,
                                    maxtime = maxtime,
                                    abstol = cache.solver_args.abstol,
                                    reltol = cache.solver_args.reltol; cache.solver_args...)

    t0 = time()
    opt_res = SpeedMapping.speedmapping(cache.u0; f = _loss, (g!) = cache.f.grad,
                                        lower = cache.lb,
                                        upper = cache.ub, opt_args...)
    t1 = time()
    opt_ret = Symbol(opt_res.converged)

    SciMLBase.build_solution(cache, cache.opt,
                             opt_res.minimizer, _loss(opt_res.minimizer);
                             original = opt_res, retcode = opt_ret, solve_time = t1 - t0)
end

end
