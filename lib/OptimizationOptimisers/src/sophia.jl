using Optimization.LinearAlgebra

struct Sophia
    lr::Float64
    betas::Tuple{Float64, Float64}
    eps::Float64
    weight_decay::Float64
    k::Integer
    rho::Float64
end

SciMLBase.supports_opt_cache_interface(opt::Sophia) = true

function Sophia(; lr = 1e-3, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 1e-1, k = 10,
    rho = 0.04)
    Sophia(lr, betas, eps, weight_decay, k, rho)
end

clip(z, ρ) = max(min(z, ρ), -ρ)

function SciMLBase.__init(prob::OptimizationProblem, opt::Sophia,
    data = Optimization.DEFAULT_DATA;
    maxiters::Number = 1000, callback = (args...) -> (false),
    progress = false, save_best = true, kwargs...)
    return OptimizationCache(prob, opt, data; maxiters, callback, progress,
        save_best, kwargs...)
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
    C,
}) where {
    F,
    RC,
    LB,
    UB,
    LC,
    UC,
    S,
    O <:
    Sophia,
    D,
    P,
    C,
}
    local x, cur, state
    uType = eltype(cache.u0)
    lr = uType(cache.opt.lr)
    betas = uType.(cache.opt.betas)
    eps = uType(cache.opt.eps)
    weight_decay = uType(cache.opt.weight_decay)
    rho = uType(cache.opt.rho)

    if cache.data != Optimization.DEFAULT_DATA
        maxiters = length(cache.data)
        data = cache.data
    else
        maxiters = Optimization._check_and_convert_maxiters(cache.solver_args.maxiters)
        data = Optimization.take(cache.data, maxiters)
    end

    maxiters = Optimization._check_and_convert_maxiters(maxiters)

    _loss = function (θ)
        if isnothing(cache.callback) && isnothing(data)
            return first(cache.f(θ, cache.p))
        elseif isnothing(cache.callback)
            return first(cache.f(θ, cache.p, cur...))
        elseif isnothing(data)
            x = cache.f(θ, cache.p)
            return first(x)
        else
            x = cache.f(θ, cache.p, cur...)
            return first(x)
        end
    end
    f = cache.f
    θ = copy(cache.u0)
    gₜ = zero(θ)
    mₜ = zero(θ)
    hₜ = zero(θ)
    for (i, d) in enumerate(data)
        f.grad(gₜ, θ, d...)
        x = cache.f(θ, cache.p, d...)
        cb_call = cache.callback(θ, x...)
        if !(typeof(cb_call) <: Bool)
            error("The callback should return a boolean `halt` for whether to stop the optimization process. Please see the sciml_train documentation for information.")
        elseif cb_call
            break
        end
        mₜ = betas[1] .* mₜ + (1 - betas[1]) .* gₜ

        if i % cache.opt.k == 1
            hₜ₋₁ = copy(hₜ)
            u = randn(uType, length(θ))
            f.hv(hₜ, θ, u, d...)
            hₜ = betas[2] .* hₜ₋₁ + (1 - betas[2]) .* (u .* hₜ)
        end
        θ = θ .- lr * weight_decay .* θ
        θ = θ .-
            lr .* clip.(mₜ ./ max.(hₜ, Ref(eps)), Ref(rho))
    end

    return SciMLBase.build_solution(cache, cache.opt,
        θ,
        x)
end
