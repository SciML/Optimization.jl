module OptimizationOptimisers

using Reexport, Printf, ProgressLogging
@reexport using Optimisers, Optimization
using Optimization.SciMLBase, MLUtils

SciMLBase.supports_opt_cache_interface(opt::AbstractRule) = true
SciMLBase.requiresgradient(opt::AbstractRule) = true
SciMLBase.allowsfg(opt::AbstractRule) = true

function SciMLBase.__init(
        prob::SciMLBase.OptimizationProblem, opt::AbstractRule; save_best = true,
        callback = (args...) -> (false), epochs = nothing,
        progress = false, kwargs...)
    return OptimizationCache(prob, opt; save_best, callback, progress, epochs,
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
        AbstractRule,
        D,
        P,
        C
}
    maxiters = if cache.solver_args.epochs === nothing
        if cache.solver_args.maxiters === nothing
            throw(ArgumentError("The number of epochs must be specified with either the epochs or maxiters kwarg."))
        else
            cache.solver_args.maxiters
        end
    else
        cache.solver_args.epochs
    end

    maxiters = Optimization._check_and_convert_maxiters(cache.solver_args.maxiters)
    if maxiters === nothing
        throw(ArgumentError("The number of epochs must be specified as the epochs or maxiters kwarg."))
    end

    if cache.p isa MLUtils.DataLoader
        data = cache.p
        dataiterate = true
    else
        data = [cache.p]
        dataiterate = false
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
    Optimization.@withprogress cache.progress name="Training" begin
        for _ in 1:maxiters
            for (i, d) in enumerate(data)
                if cache.f.fg !== nothing && dataiterate
                    x = cache.f.fg(G, θ, d)
                elseif dataiterate
                    cache.f.grad(G, θ, d)
                    x = cache.f(θ, d)
                elseif cache.f.fg !== nothing
                    x = cache.f.fg(G, θ)
                else
                    cache.f.grad(G, θ)
                    x = cache.f(θ)
                end
                opt_state = Optimization.OptimizationState(iter = i,
                    u = θ,
                    objective = x[1],
                    grad = G,
                    original = state)
                cb_call = cache.callback(opt_state, x...)
                if !(cb_call isa Bool)
                    error("The callback should return a boolean `halt` for whether to stop the optimization process. Please see the `solve` documentation for information.")
                elseif cb_call
                    break
                end
                msg = @sprintf("loss: %.3g", first(x)[1])
                cache.progress && ProgressLogging.@logprogress msg i/maxiters

                if cache.solver_args.save_best
                    if first(x)[1] < first(min_err)[1]  #found a better solution
                        min_opt = opt
                        min_err = x
                        min_θ = copy(θ)
                    end
                    if i == maxiters  #Last iter, revert to best.
                        opt = min_opt
                        x = min_err
                        θ = min_θ
                        cache.f.grad(G, θ, d...)
                        opt_state = Optimization.OptimizationState(iter = i,
                            u = θ,
                            objective = x[1],
                            grad = G,
                            original = state)
                        cache.callback(opt_state, x...)
                        break
                    end
                end
                state, θ = Optimisers.update(state, θ, G)
            end
        end
    end

    t1 = time()
    stats = Optimization.OptimizationStats(; iterations = maxiters,
        time = t1 - t0, fevals = maxiters, gevals = maxiters)
    SciMLBase.build_solution(cache, cache.opt, θ, first(x)[1], stats = stats)
    # here should be build_solution to create the output message
end

end
