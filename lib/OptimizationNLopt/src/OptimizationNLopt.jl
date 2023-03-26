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

SciMLBase.supports_opt_cache_interface(opt::Union{NLopt.Algorithm, NLopt.Opt}) = true
SciMLBase.has_reinit(cache::NLoptOptimizationCache) = true
function SciMLBase.reinit!(cache::NLoptOptimizationCache; p = missing, u0 = missing)
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
        opt.upper_bounds = cache.ub
    end

    if cache.lb !== nothing
        opt.lower_bounds = cache.lb
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

function __nlopt_status_to_ReturnCode(status::Symbol)
    if status in Symbol.([
                             NLopt.SUCCESS,
                             NLopt.STOPVAL_REACHED,
                             NLopt.FTOL_REACHED,
                             NLopt.XTOL_REACHED,
                             NLopt.ROUNDOFF_LIMITED,
                         ])
        return ReturnCode.Success
    elseif status == Symbol(NLopt.MAXEVAL_REACHED)
        return ReturnCode.MaxIters
    elseif status == Symbol(NLopt.MAXTIME_REACHED)
        return ReturnCode.MaxTime
    elseif status in Symbol.([
                                 NLopt.OUT_OF_MEMORY,
                                 NLopt.INVALID_ARGS,
                                 NLopt.FAILURE,
                                 NLopt.FORCED_STOP,
                             ])
        return ReturnCode.Failure
    else
        return ReturnCode.Default
    end
end

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
    retcode = __nlopt_status_to_ReturnCode(ret)
    
    if retcode == ReturnCode.Failure
        @warn "NLopt failed to converge: $(ret)"
        minx = fill(NaN, length(cache.u0))
        minf = NaN
    end
    SciMLBase.build_solution(cache, cache.opt, minx,
                             minf; original = opt_setup, retcode = retcode,
                             solve_time = t1 - t0)
end

end
