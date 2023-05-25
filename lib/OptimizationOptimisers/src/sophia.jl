using Optimization.LinearAlgebra
struct Sophia
    lr
    betas
    eps
    weight_decay
    k
    estimator
    rho
end

Sophia(; lr = 1e-3, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 1e-1, k = 10,
       estimator = "Hutchinson", rho = 0.04) = Sophia(lr, betas, eps, weight_decay, k, estimator, rho)

clip(z, ρ) = max(min(z,ρ), -ρ)

function SciMLBase.__solve(prob::OptimizationProblem, opt::Sophia, data = Optimization.DEFAULT_DATA; progress = false, maxiters = 1000)
    local x, cur, state

    if data != Optimization.DEFAULT_DATA
        maxiters = length(data)
        data = data
    else
        maxiters = Optimization._check_and_convert_maxiters(maxiters)
        data = Optimization.take(data, maxiters)
    end

    maxiters = Optimization._check_and_convert_maxiters(maxiters)

    _loss = function (θ)
        if isnothing(callback) && isnothing(data)
            return first(prob.f(θ, prob.p))
        elseif isnothing(callback)
            return first(prob.f(θ, prob.p, cur...))
        elseif isnothing(data)
            x = prob.f(θ, prob.p)
            return first(x)
        else
            x = prob.f(θ, prob.p, cur...)
            return first(x)
        end
    end
    f = Optimization.instantiate_function(prob.f, prob.u0, prob.f.adtype, prob.p)
    θ = copy(prob.u0)
    gₜ = zero(θ)
    mₜ = zero(θ)
    hₜ = zero(θ)
    for (i, d) in enumerate(data)
        f.grad(gₜ, θ, d...)
        mₜ = opt.betas[1] .* mₜ + (1 - opt.betas[1]) .* gₜ
        if i % opt.k == 1
            hₜ₋₁ = copy(hₜ)
            u = randn(length(θ))
            f.hv(hₜ, θ, u, d...)
            hₜ = opt.betas[2] .* hₜ₋₁ + (1 - opt.betas[2]) .* ( u .* hₜ)
        end
        θ  = θ .- opt.lr*opt.weight_decay .* θ
        θ = θ .- opt.lr .* clip.(mₜ ./ max.(hₜ, Ref(opt.eps)), Ref(opt.rho))
        # println(θ)
        # println(gₜ)
        # println(hₜ)
    end

    SciMLBase.build_solution(SciMLBase.DefaultOptimizationCache(prob.f, prob.p), opt,
                             θ,
                             prob.f(θ, prob.p))
end
