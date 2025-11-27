module OptimizationSophia

using Reexport
using SciMLBase
using OptimizationBase: OptimizationCache
@reexport using OptimizationBase
using Random

"""
    Sophia(; η = 1e-3, βs = (0.9, 0.999), ϵ = 1e-8, λ = 1e-1, k = 10, ρ = 0.04)

A second-order optimizer that incorporates diagonal Hessian information for faster convergence.

Based on the paper "Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training"
(https://arxiv.org/abs/2305.14342). Sophia uses an efficient estimate of the diagonal of the Hessian
matrix to adaptively adjust the learning rate for each parameter, achieving faster convergence than
first-order methods like Adam and SGD while avoiding the computational cost of full second-order methods.

## Arguments

  - `η::Float64 = 1e-3`: Learning rate (step size)
  - `βs::Tuple{Float64, Float64} = (0.9, 0.999)`: Exponential decay rates for the first moment (β₁)
    and diagonal Hessian (β₂) estimates
  - `ϵ::Float64 = 1e-8`: Small constant for numerical stability
  - `λ::Float64 = 1e-1`: Weight decay coefficient for L2 regularization
  - `k::Integer = 10`: Frequency of Hessian diagonal estimation (every k iterations)
  - `ρ::Float64 = 0.04`: Clipping threshold for the update to maintain stability

## Example

```julia
using OptimizationBase, OptimizationSophia

# Define optimization problem
rosenbrock(x, p) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
x0 = zeros(2)
optf = OptimizationFunction(rosenbrock, OptimizationBase.AutoZygote())
prob = OptimizationProblem(optf, x0)

# Solve with Sophia
sol = solve(prob, Sophia(η = 0.01, k = 5))
```

## Notes

Sophia is particularly effective for:

  - Large-scale optimization problems
  - Neural network training
  - Problems where second-order information can significantly improve convergence

The algorithm maintains computational efficiency by only estimating the diagonal of the Hessian
matrix using a Hutchinson trace estimator with random vectors, making it more scalable than
full second-order methods while still leveraging curvature information.
"""
struct Sophia
    η::Float64
    βs::Tuple{Float64, Float64}
    ϵ::Float64
    λ::Float64
    k::Integer
    ρ::Float64
end

SciMLBase.has_init(opt::Sophia) = true
SciMLBase.allowscallback(opt::Sophia) = true
SciMLBase.requiresgradient(opt::Sophia) = true
SciMLBase.allowsfg(opt::Sophia) = true
SciMLBase.requireshessian(opt::Sophia) = true

function Sophia(; η = 1e-3, βs = (0.9, 0.999), ϵ = 1e-8, λ = 1e-1, k = 10,
        ρ = 0.04)
    Sophia(η, βs, ϵ, λ, k, ρ)
end

clip(z, ρ) = max(min(z, ρ), -ρ)

function SciMLBase.__init(prob::OptimizationProblem, opt::Sophia;
        maxiters::Number = 1000, callback = (args...) -> (false),
        progress = false, save_best = true, kwargs...)
    return OptimizationCache(prob, opt; maxiters, callback, progress,
        save_best, kwargs...)
end

function SciMLBase.__solve(cache::OptimizationCache{O}) where {O <: Sophia}
    local x, cur, state
    uType = eltype(cache.u0)
    η = uType(cache.opt.η)
    βs = uType.(cache.opt.βs)
    ϵ = uType(cache.opt.ϵ)
    λ = uType(cache.opt.λ)
    ρ = uType(cache.opt.ρ)

    maxiters = OptimizationBase._check_and_convert_maxiters(cache.solver_args.maxiters)

    if OptimizationBase.isa_dataiterator(cache.p)
        data = cache.p
        dataiterate = true
    else
        data = [cache.p]
        dataiterate = false
    end

    f = cache.f
    θ = copy(cache.u0)
    gₜ = zero(θ)
    mₜ = zero(θ)
    hₜ = zero(θ)
    for epoch in 1:maxiters
        for (i, d) in enumerate(data)
            if cache.f.fg !== nothing && dataiterate
                x = cache.f.fg(gₜ, θ, d)
            elseif dataiterate
                cache.f.grad(gₜ, θ, d)
                x = cache.f(θ, d)
            elseif cache.f.fg !== nothing
                x = cache.f.fg(gₜ, θ)
            else
                cache.f.grad(gₜ, θ)
                x = cache.f(θ)
            end
            opt_state = OptimizationBase.OptimizationState(;
                iter = i + (epoch - 1) * length(data),
                u = θ,
                objective = first(x),
                grad = gₜ,
                original = nothing,
                p = d)
            cb_call = cache.callback(opt_state, x...)
            if !(cb_call isa Bool)
                error("The callback should return a boolean `halt` for whether to stop the optimization process. Please see the sciml_train documentation for information.")
            elseif cb_call
                break
            end
            mₜ = βs[1] .* mₜ + (1 - βs[1]) .* gₜ

            if i % cache.opt.k == 1
                hₜ₋₁ = copy(hₜ)
                u = similar(θ)
                randn!(u)
                f.hv(hₜ, θ, u, d)
                hₜ = βs[2] .* hₜ₋₁ + (1 - βs[2]) .* (u .* hₜ)
            end
            θ = θ .- η * λ .* θ
            θ = θ .-
                η .* clip.(mₜ ./ max.(hₜ, Ref(ϵ)), Ref(ρ))
        end
    end

    return SciMLBase.build_solution(cache, cache.opt,
        θ,
        x, retcode = ReturnCode.Success)
end

end
