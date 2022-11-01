module OptimizationGCMAES

using Reexport
@reexport using Optimization
using GCMAES, Optimization.SciMLBase

export GCMAESOpt

struct GCMAESOpt end

SciMLBase.requiresbounds(::GCMAESOpt) = true
SciMLBase.allowsbounds(::GCMAESOpt) = true

function __map_optimizer_args(cache::GCMAESOptimizationCache, opt::GCMAESOpt;
                              callback = nothing,
                              maxiters::Union{Number, Nothing} = nothing,
                              maxtime::Union{Number, Nothing} = nothing,
                              abstol::Union{Number, Nothing} = nothing,
                              reltol::Union{Number, Nothing} = nothing)

    # add optimiser options from kwargs
    mapped_args = (;)

    if !(isnothing(maxiters))
        mapped_args = (; mapped_args..., maxiter = maxiters)
    end

    if !(isnothing(maxtime))
        @warn "common maxtime is currently not used by $(opt)"
    end

    if !isnothing(abstol)
        @warn "common abstol is currently not used by $(opt)"
    end

    if !isnothing(reltol)
        @warn "common reltol is currently not used by $(opt)"
    end

    return mapped_args
end

struct GCMAESOptimizationCache{F <: OptimizationFunction, RC, LB, UB, S, O} <: SciMLBase.AbstractOptimizationCache
    f::F
    reinit_cache::RC
    lb::LB
    ub::UB
    sense::S
    opt::O
    solver_args::NamedTuple
end

function Base.getproperty(cache::GCMAESOptimizationCache, x::Symbol)
    if x in fieldnames(Optimization.ReInitCache)
        return getfield(cache.reinit_cache, x)
    end
    return getfield(cache, x)
end

function GCMAESOptimizationCache(prob::OptimizationProblem, opt; kwargs...)
    reinit_cache = Optimization.ReInitCache(prob.u0, prob.p) # everything that can be changed via `reinit`
    f = Optimization.instantiate_function(prob.f, reinit_cache, prob.f.adtype)
    return GCMAESOptimizationCache(f, reinit_cache, prob.lb, prob.ub, prob.sense, opt, NamedTuple(kwargs))
end

SciMLBase.supports_opt_cache_interface(opt::GCMAESOpt) = true

function SciMLBase.__init(prob::OptimizationProblem, opt::GCMAESOpt;
    maxiters::Union{Number, Nothing} = nothing,
    maxtime::Union{Number, Nothing} = nothing,
    abstol::Union{Number, Nothing} = nothing,
    reltol::Union{Number, Nothing} = nothing,
    progress = false,
    σ0 = 0.2,
    kwargs...)
    return GCMAESOptimizationCache(prob, opt; maxiters, maxtime, abstol, reltol, progress, σ0, kwargs...)
end

function SciMLBase.__solve(cache::GCMAESOptimizationCache)
    local x
    local G = similar(cache.u0)

    maxiters = Optimization._check_and_convert_maxiters(cache.solver_args.maxiters)
    maxtime = Optimization._check_and_convert_maxtime(cache.solver_args.maxtime)

    _loss = function (θ)
        x = cache.f.f(θ, cache.p)
        return x[1]
    end

    if !isnothing(cache.f.grad)
        g = function (θ)
            cache.f.grad(G, θ)
            return G
        end
    end

    opt_args = __map_optimizer_args(cache, opt, maxiters = cache.solver_args.maxiters, maxtime = cache.solver_args.maxtime,
                                    abstol = cache.solver_args.abstol, reltol = cache.solver_args.reltol; cache.solver_args...)

    t0 = time()
    if cache.sense === Optimization.MaxSense
        opt_xmin, opt_fmin, opt_ret = GCMAES.maximize(isnothing(cache.f.grad) ? _loss :
                                                      (_loss, g), cache.u0, cache.solver_args.σ0, cache.lb,
                                                      cache.ub; opt_args...)
    else
        opt_xmin, opt_fmin, opt_ret = GCMAES.minimize(isnothing(f.grad) ? _loss :
                                                      (_loss, g), cache.u0, cache.solver_args.σ0, cache.lb,
                                                      cache.ub; opt_args...)
    end
    t1 = time()

    SciMLBase.build_solution(cache, cache.opt,
                             opt_xmin, opt_fmin; retcode = Symbol(Bool(opt_ret)),
                             solve_time = t1 - t0)
end

end
