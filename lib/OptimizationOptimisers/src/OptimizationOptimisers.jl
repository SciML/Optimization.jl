module OptimizationOptimisers

using Reexport, Logging
@reexport using Optimisers, OptimizationBase
using SciMLBase

SciMLBase.has_init(opt::AbstractRule) = true
SciMLBase.requiresgradient(opt::AbstractRule) = true
SciMLBase.allowsfg(opt::AbstractRule) = true

function SciMLBase.__init(
        prob::SciMLBase.OptimizationProblem, opt::AbstractRule;
        callback = (args...) -> (false),
        epochs::Union{Number, Nothing} = nothing,
        maxiters::Union{Number, Nothing} = nothing,
        save_best::Bool = true, progress::Bool = false, kwargs...)
    return OptimizationCache(prob, opt; callback, epochs, maxiters,
        save_best, progress, kwargs...)
end

function SciMLBase.__solve(cache::OptimizationCache{O}) where {O <: AbstractRule}
    if OptimizationBase.isa_dataiterator(cache.p)
        data = cache.p
        dataiterate = true
    else
        data = [cache.p]
        dataiterate = false
    end

    epochs,
    maxiters = if isnothing(cache.solver_args.maxiters) &&
                  isnothing(cache.solver_args.epochs)
        throw(ArgumentError("The number of iterations must be specified with either the epochs or maxiters kwarg. Where maxiters = epochs * length(data)."))
    elseif !isnothing(cache.solver_args.maxiters) &&
           !isnothing(cache.solver_args.epochs)
        if cache.solver_args.maxiters == cache.solver_args.epochs * length(data)
            cache.solver_args.epochs, cache.solver_args.maxiters
        else
            throw(ArgumentError("Both maxiters and epochs were passed but maxiters != epochs * length(data)."))
        end
    elseif isnothing(cache.solver_args.maxiters)
        cache.solver_args.epochs, cache.solver_args.epochs * length(data)
    elseif isnothing(cache.solver_args.epochs)
        cache.solver_args.maxiters / length(data), cache.solver_args.maxiters
    end
    epochs = OptimizationBase._check_and_convert_maxiters(epochs)
    maxiters = OptimizationBase._check_and_convert_maxiters(maxiters)

    # At this point, both of them should be fine; but, let's assert it.
    @assert (!isnothing(epochs)&&!isnothing(maxiters) &&
             (maxiters == epochs * length(data))) "The number of iterations must be specified with either the epochs or maxiters kwarg. Where maxiters = epochs * length(data)."

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
    progress_id = :OptimizationOptimizersJL
    for epoch in 1:epochs, d in data
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
        opt_state = OptimizationBase.OptimizationState(
            iter = iterations,
            u = θ,
            p = d,
            objective = x[1],
            grad = G,
            original = state)
        breakall = cache.callback(opt_state, x...)
        if !(breakall isa Bool)
            error("The callback should return a boolean `halt` for whether to stop the optimization process. Please see the `solve` documentation for information.")
        elseif breakall
            break
        end
        if cache.progress
            message = "Loss: $(round(first(first(x)); digits = 3))"
            @logmsg(LogLevel(-1), "Optimization", _id=progress_id,
                message=message, progress=iterations / maxiters)
        end
        if cache.solver_args.save_best
            if first(x)[1] < first(min_err)[1]  #found a better solution
                min_opt = opt
                min_err = x
                min_θ = copy(θ)
            end
            if iterations == length(data) * epochs  #Last iter, revert to best.
                opt = min_opt
                x = min_err
                θ = min_θ
                cache.f.grad(G, θ, d)
                opt_state = OptimizationBase.OptimizationState(iter = iterations,
                    u = θ,
                    p = d,
                    objective = x[1],
                    grad = G,
                    original = state)
                breakall = cache.callback(opt_state, x...)
                break
            end
        end
        # Skip update if gradient contains NaN or Inf values
        has_nan_or_inf = any(.!(isfinite.(G)))
        if !has_nan_or_inf
            state, θ = Optimisers.update(state, θ, G)
        else
            @warn "Skipping parameter update due to NaN or Inf in gradients at iteration $iterations" maxlog=10
        end
    end
    cache.progress && @logmsg(LogLevel(-1), "Optimization",
        _id=progress_id, message="Done", progress=1.0)
    t1 = time()
    stats = OptimizationBase.OptimizationStats(; iterations,
        time = t1 - t0, fevals, gevals)
    SciMLBase.build_solution(cache, cache.opt, θ, first(x)[1], stats = stats)
end

end
