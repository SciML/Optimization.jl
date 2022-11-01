module OptimizationNLopt

using Reexport
@reexport using NLopt, Optimization
using Optimization.SciMLBase

(f::NLopt.Algorithm)() = f

SciMLBase.allowsbounds(opt::Union{NLopt.Algorithm, NLopt.Opt}) = true

struct NLoptOptimizationCache{F <: OptimizationFunction, RC, LB, UB, S, O, P} <:
       SciMLBase.AbstractOptimizationCache
    f::F
    reinit_cache::RC
    lb::LB
    ub::UB
    sense::S
    opt::O
    progress::P
    solver_args::NamedTuple
end

function Base.getproperty(cache::NLoptOptimizationCache, x::Symbol)
    if x in fieldnames(Optimization.ReInitCache)
        return getfield(cache.reinit_cache, x)
    end
    return getfield(cache, x)
end

function NLoptOptimizationCache(prob::OptimizationProblem, opt; progress, kwargs...)
    reinit_cache = Optimization.ReInitCache(prob.u0, prob.p) # everything that can be changed via `reinit`
    f = Optimization.instantiate_function(prob.f, reinit_cache, prob.f.adtype)

    return NLoptOptimizationCache(f, reinit_cache, prob.lb, prob.ub, prob.sense, opt,
                                  progress,
                                  NamedTuple(kwargs))
end

function __map_optimizer_args!(cache::NLoptOptimizationCache, opt::NLopt.Opt;
                               callback = nothing,
                               maxiters::Union{Number, Nothing} = nothing,
                               maxtime::Union{Number, Nothing} = nothing,
                               abstol::Union{Number, Nothing} = nothing,
                               reltol::Union{Number, Nothing} = nothing,
                               local_method::Union{NLopt.Algorithm, NLopt.Opt, Nothing} = nothing,
                               local_maxiters::Union{Number, Nothing} = nothing,
                               local_maxtime::Union{Number, Nothing} = nothing,
                               local_options::Union{NamedTuple, Nothing} = nothing,
                               kwargs...)
    if local_method !== nothing
        if isa(local_method, NLopt.Opt)
            if ndims(local_method) != length(cache.u0)
                error("Passed local NLopt.Opt optimization dimension does not match OptimizationProblem dimension.")
            end
            local_meth = local_method
        else
            local_meth = NLopt.Opt(local_method, length(cache.u0))
        end

        if !isnothing(local_options)
            for j in Dict(pairs(local_options))
                eval(Meta.parse("NLopt." * string(j.first) * "!"))(local_meth, j.second)
            end
        end

        if !(isnothing(local_maxiters))
            NLopt.maxeval!(local_meth, local_maxiters)
        end

        if !(isnothing(local_maxtime))
            NLopt.maxtime!(local_meth, local_maxtime)
        end

        NLopt.local_optimizer!(opt, local_meth)
    end

    # add optimiser options from kwargs
    for j in kwargs
        eval(Meta.parse("NLopt." * string(j.first) * "!"))(opt, j.second)
    end

    if cache.ub !== nothing
        NLopt.upper_bounds!(opt, cache.ub)
    end

    if cache.lb !== nothing
        NLopt.lower_bounds!(opt, cache.lb)
    end

    if !(isnothing(maxiters))
        NLopt.maxeval!(opt, maxiters)
    end

    if !(isnothing(maxtime))
        NLopt.maxtime!(opt, maxtime)
    end

    if !isnothing(abstol)
        NLopt.ftol_abs!(opt, abstol)
    end
    if !isnothing(reltol)
        NLopt.ftol_rel!(opt, reltol)
    end

    return nothing
end

SciMLBase.supports_opt_cache_interface(opt::Union{NLopt.Algorithm, NLopt.Opt}) = true

function SciMLBase.__init(prob::OptimizationProblem, opt::Union{NLopt.Algorithm, NLopt.Opt};
                          maxiters::Union{Number, Nothing} = nothing,
                          maxtime::Union{Number, Nothing} = nothing,
                          local_method::Union{NLopt.Algorithm, NLopt.Opt, Nothing} = nothing,
                          local_maxiters::Union{Number, Nothing} = nothing,
                          local_maxtime::Union{Number, Nothing} = nothing,
                          local_options::Union{NamedTuple, Nothing} = nothing,
                          abstol::Union{Number, Nothing} = nothing,
                          reltol::Union{Number, Nothing} = nothing,
                          progress = false,
                          callback = (args...) -> (false),
                          kwargs...)
    maxiters = Optimization._check_and_convert_maxiters(maxiters)
    maxtime = Optimization._check_and_convert_maxtime(maxtime)
    local_maxiters = Optimization._check_and_convert_maxiters(local_maxiters)
    local_maxtime = Optimization._check_and_convert_maxtime(local_maxtime)

    return NLoptOptimizationCache(prob, opt; maxiters, maxtime, local_method,
                                  local_maxiters, local_maxtime, local_options, abstol,
                                  reltol, progress, callback)
end

function SciMLBase.__solve(cache::NLoptOptimizationCache)
    local x

    _loss = function (θ)
        x = cache.f.f(θ, cache.p)
        cache.solver_args.callback(θ, x...)
        return x[1]
    end

    fg! = function (θ, G)
        if length(G) > 0
            cache.f.grad(G, θ)
        end

        return _loss(θ)
    end

    opt_setup = if isa(cache.opt, NLopt.Opt)
        if ndims(cache.opt) != length(cache.u0)
            error("Passed NLopt.Opt optimization dimension does not match OptimizationProblem dimension.")
        end
        cache.opt
    else
        NLopt.Opt(cache.opt, length(cache.u0))
    end

    if cache.sense === Optimization.MaxSense
        NLopt.max_objective!(opt_setup, fg!)
    else
        NLopt.min_objective!(opt_setup, fg!)
    end

    __map_optimizer_args!(cache, opt_setup, maxiters = cache.solver_args.maxiters,
                          maxtime = cache.solver_args.maxtime,
                          abstol = cache.solver_args.abstol,
                          reltol = cache.solver_args.reltol,
                          local_method = cache.solver_args.local_method,
                          local_maxiters = cache.solver_args.local_maxiters,
                          local_options = cache.solver_args.local_options;
                          cache.solver_args...)

    t0 = time()
    (minf, minx, ret) = NLopt.optimize(opt_setup, cache.u0)
    t1 = time()

    SciMLBase.build_solution(cache, cache.opt, minx,
                             minf; original = opt_setup, retcode = ret,
                             solve_time = t1 - t0)
end

end
