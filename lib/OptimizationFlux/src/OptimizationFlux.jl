module OptimizationFlux

using Reexport, Printf, ProgressLogging
@reexport using Flux, Optimization
using Optimization.SciMLBase

SciMLBase.supports_opt_cache_interface(opt::Flux.Optimise.AbstractOptimiser) = true

function SciMLBase.__init(prob::SciMLBase.OptimizationProblem,
        opt::Flux.Optimise.AbstractOptimiser,
        data = Optimization.DEFAULT_DATA; save_best = true,
        callback = (args...) -> (false),
        progress = false, kwargs...)
    return OptimizationCache(prob, opt, data; save_best, callback, progress,
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
        Flux.Optimise.AbstractOptimiser,
        D,
        P,
        C
}
    local i
    if cache.data != Optimization.DEFAULT_DATA
        maxiters = length(cache.data)
        data = cache.data
    else
        maxiters = Optimization._check_and_convert_maxiters(cache.solver_args.maxiters)
        data = Optimization.take(cache.data, maxiters)
    end

    # Flux is silly and doesn't have an abstract type on its optimizers, so assume
    # this is a Flux optimizer
    θ = copy(cache.u0)
    G = copy(θ)
    opt = deepcopy(cache.opt)

    local x, min_err, min_θ
    min_err = typemax(eltype(cache.u0)) #dummy variables
    min_opt = 1
    min_θ = cache.u0

    t0 = time()
    Optimization.@withprogress cache.progress name="Training" begin
        for (i, d) in enumerate(data)
            cache.f.grad(G, θ, d...)
            x = cache.f(θ, cache.p, d...)
            opt_state = Optimization.OptimizationState(; iter = i,
                u = θ,
                objective = x[1],
                original = opt)
            cb_call = cache.callback(opt_state, x...)
            if !(cb_call isa Bool)
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
                if i == maxiters  #Last iter, revert to best.
                    opt = min_opt
                    x = min_err
                    θ = min_θ
                    opt_state = Optimization.OptimizationState(; iter = i,
                        u = θ,
                        objective = x[1],
                        original = opt)
                    cache.callback(opt_state, x...)
                    break
                end
            end
            Flux.update!(opt, θ, G)
        end
    end

    t1 = time()
    stats = Optimization.OptimizationStats(; iterations = maxiters,
        time = t1 - t0, fevals = maxiters, gevals = maxiters)
    SciMLBase.build_solution(cache, opt, θ, x[1], stats = stats)
    # here should be build_solution to create the output message
end

end
