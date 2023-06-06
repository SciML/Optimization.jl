module OptimizationOptimisers

using Reexport, Printf, ProgressLogging
@reexport using Optimisers, Optimization
using Optimization.SciMLBase

const OptimisersOptimizers = Union{Descent, Adam, Momentum, Nesterov, RMSProp,
                                   AdaGrad, AdaMax, AdaDelta, AMSGrad, NAdam, RAdam, OAdam,
                                   AdaBelief,
                                   WeightDecay, ClipGrad, ClipNorm, OptimiserChain}

SciMLBase.supports_opt_cache_interface(opt::OptimisersOptimizers) = true

function SciMLBase.__init(prob::SciMLBase.OptimizationProblem, opt::OptimisersOptimizers,
                          data = Optimization.DEFAULT_DATA; save_best = true, kwargs...)
    return OptimizationCache(prob, opt, data; save_best,
                             kwargs...)
end

function SciMLBase.__solve(cache::OptimizationCache)
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
    Optimization.@withprogress cache.progress name="Training" begin for (i, d) in enumerate(data)
        cache.f.grad(G, θ, d...)
        x = cache.f(θ, cache.p, d...)
        cb_call = cache.callback(θ, x...)
        if !(typeof(cb_call) <: Bool)
            error("The callback should return a boolean `halt` for whether to stop the optimization process. Please see the sciml_train documentation for information.")
        elseif cb_call
            break
        end
        msg = @sprintf("loss: %.3g", x[1])
        cache.progress && ProgressLogging.@logprogress msg i/maxiters

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
                cache.callback(θ, x...)
                break
            end
        end
        state, θ = Optimisers.update(state, θ, G)
    end end

    t1 = time()

    SciMLBase.build_solution(cache, cache.opt, θ, x[1], solve_time = t1 - t0)
    # here should be build_solution to create the output message
end

end
