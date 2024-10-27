module OptimizationOptimisers

using Reexport, Printf, ProgressLogging
@reexport using Optimisers, Optimization
using Optimization.SciMLBase, Optimization.OptimizationBase

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
    if OptimizationBase.isa_dataiterator(cache.p)
        data = cache.p
        dataiterate = true
    else
        data = [cache.p]
        dataiterate = false
    end

    epochs = if cache.solver_args.epochs === nothing
        if cache.solver_args.maxiters === nothing
            throw(ArgumentError("The number of iterations must be specified with either the epochs or maxiters kwarg. Where maxiters = epochs*length(data)."))
        else
            cache.solver_args.maxiters / length(data)
        end
    else
        cache.solver_args.epochs
    end

    epochs = Optimization._check_and_convert_maxiters(epochs)
    if epochs === nothing
        throw(ArgumentError("The number of iterations must be specified with either the epochs or maxiters kwarg. Where maxiters = epochs*length(data)."))
    end

    opt = cache.opt
    θ = copy(cache.u0)
    G = copy(θ)

    local x, min_err, min_θ
    min_err = typemax(eltype(real(cache.u0))) #dummy variables
    min_opt = 1
    min_θ = cache.u0

    state = Optimisers.setup(opt, θ)
    iterations = 0
    fevals = 0
    gevals = 0
    t0 = time()
    breakall = false
    Optimization.@withprogress cache.progress name="Training" begin
        for epoch in 1:epochs
            if breakall
                break
            end
            for (i, d) in enumerate(data)
                if cache.f.fg !== nothing && dataiterate
                    x = cache.f.fg(G, θ, d)
                    iterations += 1
                    fevals += 1
                    gevals += 1
                elseif dataiterate
                    cache.f.grad(G, θ, d)
                    x = cache.f(θ, d)
                    iterations += 1
                    fevals += 2
                    gevals += 1
                elseif cache.f.fg !== nothing
                    x = cache.f.fg(G, θ)
                    iterations += 1
                    fevals += 1
                    gevals += 1
                else
                    cache.f.grad(G, θ)
                    x = cache.f(θ)
                    iterations += 1
                    fevals += 2
                    gevals += 1
                end
                opt_state = Optimization.OptimizationState(
                    iter = i + (epoch - 1) * length(data),
                    u = θ,
                    objective = x[1],
                    grad = G,
                    original = state)
                breakall = cache.callback(opt_state, x...)
                if !(breakall isa Bool)
                    error("The callback should return a boolean `halt` for whether to stop the optimization process. Please see the `solve` documentation for information.")
                elseif breakall
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
                    if i == length(data)*epochs  #Last iter, revert to best.
                        opt = min_opt
                        x = min_err
                        θ = min_θ
                        cache.f.grad(G, θ, d)
                        opt_state = Optimization.OptimizationState(iter = i,
                            u = θ,
                            objective = x[1],
                            grad = G,
                            original = state)
                        breakall = cache.callback(opt_state, x...)
                        break
                    end
                end
                state, θ = Optimisers.update(state, θ, G)
            end
        end
    end

    t1 = time()
    stats = Optimization.OptimizationStats(; iterations,
        time = t1 - t0, fevals, gevals)
    SciMLBase.build_solution(cache, cache.opt, θ, first(x)[1], stats = stats)
end

end
