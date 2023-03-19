module OptimizationOptimisers

using Reexport, Printf, ProgressLogging
@reexport using Optimisers, Optimization
using Optimization.SciMLBase

const OptimisersOptimizers = Union{Descent, Adam, Momentum, Nesterov, RMSProp,
                                   AdaGrad, AdaMax, AdaDelta, AMSGrad, NAdam, RAdam, OAdam,
                                   AdaBelief,
                                   WeightDecay, ClipGrad, ClipNorm, OptimiserChain}

struct OptimisersOptimizationCache{F <: OptimizationFunction, RC, LB, UB, S, D, O} <:
       SciMLBase.AbstractOptimizationCache
    f::F
    reinit_cache::RC
    lb::LB
    ub::UB
    sense::S
    opt::O
    data::D
    solver_args::NamedTuple
end

function Base.getproperty(cache::OptimisersOptimizationCache, x::Symbol)
    if x in fieldnames(Optimization.ReInitCache)
        return getfield(cache.reinit_cache, x)
    end
    return getfield(cache, x)
end

function OptimisersOptimizationCache(prob::OptimizationProblem, opt, data; kwargs...)
    reinit_cache = Optimization.ReInitCache(prob.u0, prob.p) # everything that can be changed via `reinit`
    f = Optimization.instantiate_function(prob.f, reinit_cache, prob.f.adtype)
    return OptimisersOptimizationCache(f, reinit_cache, prob.lb, prob.ub, prob.sense, opt,
                                       data, NamedTuple(kwargs))
end

SciMLBase.supports_opt_cache_interface(opt::OptimisersOptimizers) = true
SciMLBase.has_reinit(cache::OptimisersOptimizationCache) = true
function SciMLBase.reinit!(cache::OptimisersOptimizationCache; p = missing, u0 = missing)
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

function SciMLBase.__init(prob::OptimizationProblem, opt::OptimisersOptimizers,
                          data = Optimization.DEFAULT_DATA;
                          maxiters::Number = 1, callback = (args...) -> (false),
                          progress = false, save_best = true, kwargs...)
    return OptimisersOptimizationCache(prob, opt, data; maxiters, callback, progress,
                                       save_best, kwargs...)
end

function SciMLBase.__solve(cache::OptimisersOptimizationCache)
    if cache.data != Optimization.DEFAULT_DATA
        maxiters = length(cache.data)
        data = cache.data
    else
        maxiters = Optimization._check_and_convert_maxiters(cache.solver_args.maxiters)
        data = Optimization.take(cache.data, maxiters)
    end
    opt = cache.opt
    θ = copy(cache.u0)
    G = copy(θ)

    local x, min_err, min_θ
    min_err = typemax(eltype(real(cache.u0))) #dummy variables
    min_opt = 1
    min_θ = cache.u0

    state = Optimisers.setup(opt, θ)

    t0 = time()
    Optimization.@withprogress cache.solver_args.progress name="Training" begin for (i, d) in enumerate(data)
        cache.f.grad(G, θ, d...)
        x = cache.f.f(θ, cache.p, d...)
        cb_call = cache.solver_args.callback(θ, x...)
        if !(typeof(cb_call) <: Bool)
            error("The callback should return a boolean `halt` for whether to stop the optimization process. Please see the sciml_train documentation for information.")
        elseif cb_call
            break
        end
        msg = @sprintf("loss: %.3g", x[1])
        cache.solver_args.progress && ProgressLogging.@logprogress msg i/maxiters

        if cache.solver_args.save_best
            if first(x) < first(min_err)  #found a better solution
                min_opt = opt
                min_err = x
                min_θ = copy(θ)
            end
            if i == maxiters  #Last iteration, revert to best.
                opt = min_opt
                x = min_err
                θ = min_θ
                cache.solver_args.callback(θ, x...)
                break
            end
        end
        state, θ = Optimisers.update(state, θ, G)
    end end

    t1 = time()

    SciMLBase.build_solution(cache, opt, θ, x[1], solve_time = t1 - t0)
    # here should be build_solution to create the output message
end

end
