module OptimizationGCMAES

using Reexport
@reexport using Optimization
using GCMAES, Optimization.SciMLBase

export GCMAESOpt

struct GCMAESOpt end

SciMLBase.requiresbounds(::GCMAESOpt) = true
SciMLBase.allowsbounds(::GCMAESOpt) = true
SciMLBase.allowscallback(::GCMAESOpt) = false
SciMLBase.supports_opt_cache_interface(opt::GCMAESOpt) = true
SciMLBase.requiresgradient(::GCMAESOpt) = true
SciMLBase.requireshessian(::GCMAESOpt) = false
SciMLBase.requiresconsjac(::GCMAESOpt) = false
SciMLBase.requiresconshess(::GCMAESOpt) = false


function __map_optimizer_args(cache::OptimizationCache, opt::GCMAESOpt;
        callback = nothing,
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        kwargs...)

    # add optimiser options from kwargs
    mapped_args = (;)

    if !(isnothing(maxiters))
        mapped_args = (; mapped_args..., maxiter = maxiters)
    end

    if !(isnothing(maxtime))
        mapped_args = (; mapped_args..., maxtime = maxtime)
    end

    if !isnothing(abstol)
        @warn "common abstol is currently not used by $(opt)"
    end

    if !isnothing(reltol)
        @warn "common reltol is currently not used by $(opt)"
    end

    return mapped_args
end

function SciMLBase.__init(prob::SciMLBase.OptimizationProblem,
        opt::GCMAESOpt,
        data = Optimization.DEFAULT_DATA; σ0 = 0.2,
        callback = (args...) -> (false),
        progress = false, kwargs...)
    return OptimizationCache(prob, opt, data; σ0 = σ0, callback = callback,
        progress = progress,
        kwargs...)
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
        GCMAESOpt,
        D,
        P,
        C
}
    local x
    local G = similar(cache.u0)

    _loss = function (θ)
        x = cache.f(θ, cache.p)
        return x[1]
    end

    if !isnothing(cache.f.grad)
        g = function (θ)
            cache.f.grad(G, θ)
            return G
        end
    end

    maxiters = Optimization._check_and_convert_maxiters(cache.solver_args.maxiters)
    maxtime = Optimization._check_and_convert_maxtime(cache.solver_args.maxtime)

    opt_args = __map_optimizer_args(cache, cache.opt; cache.solver_args...,
        maxiters = maxiters,
        maxtime = maxtime)

    t0 = time()
    if cache.sense === Optimization.MaxSense
        opt_xmin, opt_fmin, opt_ret = GCMAES.maximize(
            isnothing(cache.f.grad) ? _loss :
            (_loss, g), cache.u0,
            cache.solver_args.σ0, cache.lb,
            cache.ub; opt_args...)
    else
        opt_xmin, opt_fmin, opt_ret = GCMAES.minimize(
            isnothing(cache.f.grad) ? _loss :
            (_loss, g), cache.u0,
            cache.solver_args.σ0, cache.lb,
            cache.ub; opt_args...)
    end
    t1 = time()
    stats = Optimization.OptimizationStats(;
        iterations = maxiters === nothing ? 0 : maxiters,
        time = t1 - t0)
    SciMLBase.build_solution(cache, cache.opt,
        opt_xmin, opt_fmin; retcode = Symbol(Bool(opt_ret)),
        stats = stats)
end

end
