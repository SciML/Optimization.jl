module OptimizationSOAP

using Reexport, Logging, LinearAlgebra
using SciMLBase
using OptimizationBase: OptimizationCache
@reexport using Optimisers, OptimizationBase

export SOAP

"""
    SOAP(; eta=3e-3, beta=(0.95, 0.95), shampoo_beta=-1.0, epsilon=1e-8,
           freq=10, max_dim=10000, weight_decay=0.01, correct_bias=true)

SOAP optimizer (ShampoO with Adam in the Preconditioner's eigenbasis).
Runs AdamW in the eigenbasis of Shampoo's preconditioner for 2D parameters,
standard AdamW for 1D/scalar parameters.

Reference: Vyas et al., https://arxiv.org/abs/2409.11321

## Arguments

  - `eta`: learning rate
  - `beta`: (β₁, β₂) for Adam momentum and second moment
  - `shampoo_beta`: separate β for preconditioner EMA; if < 0, uses β₂
  - `epsilon`: numerical stability constant
  - `freq`: how often to recompute eigenbasis
  - `max_dim`: dimensions larger than this use identity rotation
  - `weight_decay`: decoupled weight decay, applied as `lr * wd`
  - `correct_bias`: whether to apply Adam bias correction

## Example

```julia
using OptimizationSOAP

rosenbrock(x, p) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
optf = OptimizationFunction(rosenbrock, OptimizationBase.AutoZygote())
prob = OptimizationProblem(optf, zeros(2))
sol = solve(prob, SOAP(eta = 0.003), maxiters = 1000)
```
"""
struct SOAP <: Optimisers.AbstractRule
    eta::Float64
    beta::Tuple{Float64, Float64}
    shampoo_beta::Float64
    epsilon::Float64
    freq::Int
    max_dim::Int
    weight_decay::Float64
    correct_bias::Bool
end

function SOAP(;
        eta = 3e-3, beta = (0.95, 0.95), shampoo_beta = -1.0,
        epsilon = 1e-8, freq = 10, max_dim = 10000,
        weight_decay = 0.01, correct_bias = true)
    SOAP(eta, beta, shampoo_beta, epsilon, freq, max_dim, weight_decay, correct_bias)
end

_soap_sβ(o::SOAP) = o.shampoo_beta >= 0 ? Float32(o.shampoo_beta) : Float32(o.beta[2])

@inline _soap_zeros(ref, dims...) = fill!(similar(ref, Float32, dims...), 0)

function _soap_eye(ref, n)
    A = similar(ref, Float32, n, n)
    copyto!(A, Matrix{Float32}(I, n, n))
    return A
end

_soap_f32(x::AbstractArray{Float32}) = x
_soap_f32(x::AbstractArray) = Float32.(x)

function _soap_eigh(P)
    Pc = Float64.(Array(P))
    S = Symmetric((Pc .+ Pc') ./ 2 + 1e-30 * I)
    E = eigen(S)
    Float32.(E.vectors[:, end:-1:1])
end

function _soap_power_qr(P, Q_old)
    Pf = Float32.(Array(P))
    Qf = Float32.(Array(Q_old))
    est = diag(Qf' * Pf * Qf)
    perm = sortperm(est; rev = true)
    F = qr(Pf * Qf[:, perm])
    Matrix(F.Q), perm
end

function _soap_reorder!(eas::AbstractMatrix, perm::Vector{Int}, dim::Int)
    idx = similar(eas, Int, length(perm))
    copyto!(idx, perm)
    tmp = dim == 1 ? eas[idx, :] : eas[:, idx]
    copyto!(eas, tmp)
end

_soap_proj(X, QL, QR, uL, uR) = begin Y = X; uL && (Y = QL' * Y); uR && (Y = Y * QR); Y end
_soap_back(X, QL, QR, uL, uR) = begin Y = X; uL && (Y = QL * Y); uR && (Y = Y * QR'); Y end

function _soap_accum!(state, G::AbstractMatrix, sβ, uL, uR)
    a = Float32(1 - sβ); b = Float32(sβ)
    uL && mul!(state.L, G, G', a, b)
    uR && mul!(state.R, G', G, a, b)
end

function Optimisers.init(o::SOAP, x::AbstractMatrix)
    m, n = size(x)
    uL = m <= o.max_dim; uR = n <= o.max_dim
    (exp_avg    = _soap_zeros(x, m, n),
     exp_avg_sq = _soap_zeros(x, m, n),
     L  = uL ? _soap_zeros(x, m, m) : nothing,
     R  = uR ? _soap_zeros(x, n, n) : nothing,
     QL = uL ? _soap_eye(x, m) : nothing,
     QR = uR ? _soap_eye(x, n) : nothing,
     step = Int[0], q_ready = Bool[false])
end

function Optimisers.init(o::SOAP, x::AbstractVector)
    (exp_avg = _soap_zeros(x, length(x)),
     exp_avg_sq = _soap_zeros(x, length(x)),
     step = Int[0])
end

function Optimisers.init(o::SOAP, x::AbstractArray)
    (exp_avg = fill!(similar(x, Float32), 0),
     exp_avg_sq = fill!(similar(x, Float32), 0),
     step = Int[0])
end

function Optimisers.apply!(o::SOAP, state, x::AbstractMatrix{T}, dx) where {T}
    β1, β2 = Float32.(o.beta)
    sβ = _soap_sβ(o); ε = Float32(o.epsilon)
    G = _soap_f32(dx)
    uL = !isnothing(state.QL); uR = !isnothing(state.QR)

    if !state.q_ready[1]
        _soap_accum!(state, G, sβ, uL, uR)
        uL && copyto!(state.QL, _soap_eigh(state.L))
        uR && copyto!(state.QR, _soap_eigh(state.R))
        state.q_ready[1] = true
        return state, fill!(similar(dx, T), zero(T))
    end

    state.step[1] += 1
    t = state.step[1]
    ea = state.exp_avg; eas = state.exp_avg_sq

    G_rot = _soap_proj(G, state.QL, state.QR, uL, uR)
    @. ea  = β1 * ea  + (1 - β1) * G_rot
    @. eas = β2 * eas + (1 - β2) * (G_rot * G_rot)

    denom = @. sqrt(eas) + ε
    s = T(o.eta)
    o.correct_bias && (s = T(o.eta) * T(sqrt(1 - β2^t)) / T(1 - β1^t))

    norm_grad = _soap_back(ea ./ denom, state.QL, state.QR, uL, uR)
    λ = T(o.eta) * T(o.weight_decay)
    dx_out = @. s * T(norm_grad) + λ * x

    ea_orig = _soap_back(ea, state.QL, state.QR, uL, uR)
    _soap_accum!(state, G, sβ, uL, uR)

    if t > 0 && t % o.freq == 0
        if uL
            Q_new, perm = _soap_power_qr(state.L, state.QL)
            _soap_reorder!(eas, perm, 1)
            copyto!(state.QL, Q_new)
        end
        if uR
            Q_new, perm = _soap_power_qr(state.R, state.QR)
            _soap_reorder!(eas, perm, 2)
            copyto!(state.QR, Q_new)
        end
    end

    ea .= _soap_proj(ea_orig, state.QL, state.QR, uL, uR)
    return state, dx_out
end

function Optimisers.apply!(o::SOAP, state, x::AbstractVector{T}, dx) where {T}
    β1, β2 = Float32.(o.beta); ε = Float32(o.epsilon); G = _soap_f32(dx)
    state.step[1] += 1; t = state.step[1]
    @. state.exp_avg    = β1 * state.exp_avg    + (1 - β1) * G
    @. state.exp_avg_sq = β2 * state.exp_avg_sq + (1 - β2) * G^2
    denom = @. sqrt(state.exp_avg_sq) + ε
    s = T(o.eta)
    o.correct_bias && (s = T(o.eta) * T(sqrt(1 - β2^t)) / T(1 - β1^t))
    λ = T(o.eta) * T(o.weight_decay)
    return state, @. s * T(state.exp_avg / denom) + λ * x
end

function Optimisers.apply!(o::SOAP, state, x::AbstractArray{T}, dx) where {T}
    β1, β2 = Float32.(o.beta); ε = Float32(o.epsilon); G = _soap_f32(dx)
    state.step[1] += 1; t = state.step[1]
    @. state.exp_avg    = β1 * state.exp_avg    + (1 - β1) * G
    @. state.exp_avg_sq = β2 * state.exp_avg_sq + (1 - β2) * G^2
    denom = @. sqrt(state.exp_avg_sq) + ε
    s = T(o.eta)
    o.correct_bias && (s = T(o.eta) * T(sqrt(1 - β2^t)) / T(1 - β1^t))
    λ = T(o.eta) * T(o.weight_decay)
    return state, @. s * T(state.exp_avg / denom) + λ * x
end

SciMLBase.has_init(::SOAP) = true
SciMLBase.requiresgradient(::SOAP) = true
SciMLBase.allowsfg(::SOAP) = true
SciMLBase.allowscallback(::SOAP) = true

function SciMLBase.__init(prob::OptimizationProblem, opt::SOAP;
        callback = (args...) -> (false),
        epochs::Union{Number, Nothing} = nothing,
        maxiters::Union{Number, Nothing} = nothing,
        save_best::Bool = true, progress::Bool = false, kwargs...)
    return OptimizationCache(prob, opt; callback, epochs, maxiters,
        save_best, progress, kwargs...)
end

function SciMLBase.__solve(cache::OptimizationCache{SOAP})
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

    opt = cache.opt
    θ = copy(cache.u0)
    G = copy(θ)

    local x, min_err, min_θ
    min_err = typemax(eltype(real(cache.u0)))
    min_opt = 1
    min_θ = cache.u0

    state = Optimisers.setup(opt, θ)
    iterations = 0
    fevals = 0
    gevals = 0
    t0 = time()
    breakall = false
    progress_id = :OptimizationSOAP
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
            original = state
        )
        breakall = cache.callback(opt_state, x...)
        if !(breakall isa Bool)
            error("The callback should return a boolean `halt` for whether to stop the optimization process. Please see the `solve` documentation for information.")
        elseif breakall
            break
        end
        if cache.progress
            message = "Loss: $(round(first(first(x)); digits = 3))"
            @logmsg(
                LogLevel(-1), "Optimization", _id = progress_id,
                message = message, progress = iterations / maxiters
            )
        end
        if cache.solver_args.save_best
            if first(x)[1] < first(min_err)[1]
                min_opt = opt
                min_err = x
                min_θ = copy(θ)
            end
            if iterations == length(data) * epochs
                opt = min_opt
                x = min_err
                θ = min_θ
                cache.f.grad(G, θ, d)
                opt_state = OptimizationBase.OptimizationState(
                    iter = iterations,
                    u = θ,
                    p = d,
                    objective = x[1],
                    grad = G,
                    original = state
                )
                breakall = cache.callback(opt_state, x...)
                break
            end
        end
        if all(isfinite, G)
            state, θ = Optimisers.update(state, θ, G)
        else
            @SciMLMessage(
                lazy"Skipping parameter update due to NaN or Inf in gradients at iteration $iterations",
                cache.verbose, :nan_inf_gradients
            )
        end
    end
    cache.progress && @logmsg(
        LogLevel(-1), "Optimization",
        _id = progress_id, message = "Done", progress = 1.0
    )
    t1 = time()
    stats = OptimizationBase.OptimizationStats(;
        iterations,
        time = t1 - t0, fevals, gevals
    )
    return SciMLBase.build_solution(cache, cache.opt, θ, first(x)[1], stats = stats)
end

end