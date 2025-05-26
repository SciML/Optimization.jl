module OptimizationODE

using Reexport
@reexport using Optimization
using Optimization.SciMLBase
using Optimization.LinearAlgebra
import ForwardDiff

export ODEGradientDescent, RKChebyshevDescent, RKAccelerated, PRKChebyshevDescent

# Spectrum estimation 

# Full Hessian for small n
function _full_spectrum(f, u0, p)
  g(u) = f(u,p)[1]
  H = ForwardDiff.hessian(g, u0)
  λ = eigen(H).values
  return minimum(λ), maximum(λ)
end

# Power‐method for largest eigenvalue
function _power_max(f, u0, p; iters=20)
  n = length(u0);  
  x = randn(n)
  for _ in 1:iters
    ε = 1e-6
    ∇0  = ForwardDiff.gradient(u->f(u,p)[1], u0)
    ∇1  = ForwardDiff.gradient(u->f(u,p)[1], u0 .+ ε*x)
    Hx  = (∇1 .- ∇0) ./ ε
    x   = Hx ./ norm(Hx)
  end
  # Rayleigh quotient
  ε = 1e-6
  ∇0  = ForwardDiff.gradient(u->f(u,p)[1], u0)
  ∇1  = ForwardDiff.gradient(u->f(u,p)[1], u0 .+ ε*x)
  Hx  = (∇1 .- ∇0) ./ ε
  return dot(x, Hx)
end

# Fallback for smallest eigenvalue
_power_min(f, u0, p; iters=20) = 1e-6

"""
  estimate_spectrum(f, u0, p; large_n_thresh=50)

Return (ℓ,L) either via full Hessian (n≤large_n_thresh) or power‐method (n>large_n_thresh).
"""
function estimate_spectrum(f, u0, p; large_n_thresh=50)
  n = length(u0)
  if n ≤ large_n_thresh
    return _full_spectrum(f, u0, p)
  else
    return (_power_min(f,u0,p), _power_max(f,u0,p))
  end
end

# Euler‐ODEGradientDescent 

struct ODEGradientDescent end

SciMLBase.requiresbounds(::ODEGradientDescent)         = false
SciMLBase.allowsbounds(::ODEGradientDescent)           = false
SciMLBase.allowscallback(::ODEGradientDescent)         = false
SciMLBase.supports_opt_cache_interface(::ODEGradientDescent) = true
SciMLBase.requiresgradient(::ODEGradientDescent)       = true

function SciMLBase.__init(prob::SciMLBase.OptimizationProblem,
                          opt::ODEGradientDescent,
                          data=Optimization.DEFAULT_DATA;
                          η::Float64=0.1,
                          tmax::Float64=1.0,
                          dt::Float64=0.01,
                          callback=(args...)->false,
                          progress=false,
                          kwargs...)
  return OptimizationCache(prob, opt, data;
                           η=η, tmax=tmax, dt=dt,
                           callback=callback,
                           progress=progress,
                           kwargs...)
end

function SciMLBase.__solve(
    cache::OptimizationCache{F,RC,LB,UB,LC,UC,S,O,D,P,C}
    ) where {F,RC,LB,UB,LC,UC,S,O<:ODEGradientDescent,D,P,C}
  u = copy(cache.u0)
  G = similar(u)
  η, dt, tmax = cache.solver_args.η, cache.solver_args.dt, cache.solver_args.tmax
  t=0.0; iter=0
  while t < tmax
    cache.f.grad(G, u, cache.p)
    u .-= η .* G .* dt
    t += dt; iter += 1
  end
  return SciMLBase.build_solution(cache, cache.opt, u, cache.f(u, cache.p)[1];
    retcode = ReturnCode.Success,
    stats   = Optimization.OptimizationStats(iter, t, iter, iter, 0))
end

# RKChebyshevDescent 

struct RKChebyshevDescent end

SciMLBase.requiresbounds(::RKChebyshevDescent)         = false
SciMLBase.allowsbounds(::RKChebyshevDescent)           = false
SciMLBase.allowscallback(::RKChebyshevDescent)         = false
SciMLBase.supports_opt_cache_interface(::RKChebyshevDescent) = true
SciMLBase.requiresgradient(::RKChebyshevDescent)       = true

function SciMLBase.__init(prob::SciMLBase.OptimizationProblem,
                          opt::RKChebyshevDescent,
                          data=Optimization.DEFAULT_DATA;
                          use_hessian::Bool=true,
                          η::Float64=1.0,
                          μ::Union{Tuple{Float64,Float64},Nothing}=nothing,
                          s::Int=5,
                          maxiters::Union{Number,Nothing}=nothing,
                          kwargs...)
  u0, p = prob.u0, prob.p
  if μ===nothing && use_hessian
    ℓ, L = estimate_spectrum(prob.f, u0, p)
  elseif μ!==nothing
    ℓ, L = μ
  else
    throw(ArgumentError("Must supply μ or use_hessian=true"))
  end
  return OptimizationCache(prob, opt, data;
                           η=η, μ=(ℓ,L), s=s, maxiters=maxiters,
                           kwargs...)
end

function SciMLBase.__solve(
    cache::OptimizationCache{F,RC,LB,UB,LC,UC,S,O,D,P,C}
    ) where {F,RC,LB,UB,LC,UC,S,O<:RKChebyshevDescent,D,P,C}
  u = copy(cache.u0)
  G = similar(u)
  η, (ℓ,L), s, maxit = cache.solver_args.η, cache.solver_args.μ, cache.solver_args.s, cache.solver_args.maxiters
  σ = (L - ℓ)/2
  δ = (L + ℓ)/2
  h = η/δ

  # First (Euler) stage
  cache.f.grad(G, u, cache.p)
  u_prev = copy(u)
  u_cur  = u_prev .- (h*δ/σ) .* G

  t=0.0; iter=0
  while isnothing(maxit) || iter < maxit
    for j in 2:s
      cache.f.grad(G, u_cur, cache.p)
      Tjm1 = chebyshevT(j-1, δ/σ)
      Tj   = chebyshevT(j,   δ/σ)
      μj = 2*(δ/σ)*Tjm1/Tj
      νj = 2*Tjm1/Tj
      u_new = -μj*h .* G .+ νj .* u_cur .- (νj - 1).*u_prev
      u_prev, u_cur = u_cur, u_new
    end
    t += h; iter += 1
    if !isnothing(maxit) && iter >= maxit
        break 
    end
  end

  val = cache.f(u_cur, cache.p)[1]
  return SciMLBase.build_solution(cache, cache.opt, u_cur, val;
    retcode = ReturnCode.Success,
    stats   = Optimization.OptimizationStats(iter, t, iter*s, iter*s, 0))
end

# RKAccelerated

struct RKAccelerated end

SciMLBase.requiresbounds(::RKAccelerated)         = false
SciMLBase.allowsbounds(::RKAccelerated)           = false
SciMLBase.allowscallback(::RKAccelerated)         = false
SciMLBase.supports_opt_cache_interface(::RKAccelerated) = true
SciMLBase.requiresgradient(::RKAccelerated)       = true

function SciMLBase.__init(prob::SciMLBase.OptimizationProblem,
                          opt::RKAccelerated,
                          data=Optimization.DEFAULT_DATA;
                          η::Float64=1.0,
                          p::Float64=2.0,
                          s::Int=4,
                          maxiters::Union{Number,Nothing}=nothing,
                          kwargs...)
  return OptimizationCache(prob, opt, data;
                           η=η, p=p, s=s, maxiters=maxiters,
                           kwargs...)
end

function SciMLBase.__solve(
    cache::OptimizationCache{F,RC,LB,UB,LC,UC,S,O,D,P,C}
    ) where {F,RC,LB,UB,LC,UC,S,O<:RKAccelerated,D,P,C}
  u = copy(cache.u0)
  G = similar(u)
  η, p, s, maxit = cache.solver_args.η, cache.solver_args.p, cache.solver_args.s, cache.solver_args.maxiters
  N = isnothing(maxit) ? 1000 : maxit
  base_h = N^(-1/(s+1))
  h = η * base_h

  t=0.0; iter=0
  while isnothing(maxit) || iter < maxit
    # simple RK4 on du/dt = -∇f
    k1 = similar(u); cache.f.grad(k1, u, cache.p)
    k2 = similar(u); cache.f.grad(k2, u .- 0.5h*k1, cache.p)
    k3 = similar(u); cache.f.grad(k3, u .- 0.5h*k2, cache.p)
    k4 = similar(u); cache.f.grad(k4, u .-   h*k3, cache.p)
    u .-= h*(k1 .+ 2k2 .+ 2k3 .+ k4)/6
    t += h; iter += 1
  end

  val = cache.f(u, cache.p)[1]
  return SciMLBase.build_solution(cache, cache.opt, u, val;
    retcode = ReturnCode.Success,
    stats   = Optimization.OptimizationStats(iter, t, iter*s, iter*s, 0))
end

# PRKChebyshevDescent 

struct PRKChebyshevDescent end

SciMLBase.requiresbounds(::PRKChebyshevDescent)         = false
SciMLBase.allowsbounds(::PRKChebyshevDescent)           = false
SciMLBase.allowscallback(::PRKChebyshevDescent)         = false
SciMLBase.supports_opt_cache_interface(::PRKChebyshevDescent) = true
SciMLBase.requiresgradient(::PRKChebyshevDescent)       = true

function SciMLBase.__init(prob::SciMLBase.OptimizationProblem,
                          opt::PRKChebyshevDescent,
                          data=Optimization.DEFAULT_DATA;
                          use_hessian::Bool=true,
                          η::Float64=1.0,
                          μ::Union{Tuple{Float64,Float64},Nothing}=nothing,
                          s::Int=4,
                          maxiters::Union{Number,Nothing}=nothing,
                          reestimate::Int=10,
                          kwargs...)
  u0, p = prob.u0, prob.p
  if μ===nothing && use_hessian
    ℓ, L = estimate_spectrum(prob.f, u0, p)
  elseif μ!==nothing
    ℓ, L = μ
  else
    throw(ArgumentError("Must supply μ or use_hessian=true"))
  end
  return OptimizationCache(prob, opt, data;
                           η=η, μ=(ℓ,L), s=s,
                           maxiters=maxiters,
                           reestimate=reestimate,
                           kwargs...)
end

function SciMLBase.__solve(
    cache::OptimizationCache{F,RC,LB,UB,LC,UC,S,O,D,P,C}
    ) where {F,RC,LB,UB,LC,UC,S,O<:PRKChebyshevDescent,D,P,C}
  u = copy(cache.u0)
  G = similar(u)
  η, (ℓ,L), s, maxit, reest = cache.solver_args.η, cache.solver_args.μ, cache.solver_args.s, cache.solver_args.maxiters, cache.solver_args.reestimate
  σ = (L - ℓ)/2
  δ = (L + ℓ)/2
  h = η/δ

  t=0.0; iter=0
  while isnothing(maxit) || iter < maxit
    # adaptive re-estimation
    if reest>0 && iter>0 && iter % reest == 0
      ℓ, L = estimate_spectrum(cache.f, u, cache.p)
      σ = (L - ℓ)/2; δ = (L + ℓ)/2; h = η/δ
    end

    # one Chebyshev block of s stages
    cache.f.grad(G, u, cache.p)
    u_prev = copy(u)
    u_cur  = u_prev .- (h*δ/σ) .* G
    for j in 2:s
      cache.f.grad(G, u_cur, cache.p)
      Tjm1 = chebyshevT(j-1, δ/σ)
      Tj   = chebyshevT(j,   δ/σ)
      μj = 2*(δ/σ)*Tjm1/Tj
      νj = 2*Tjm1/Tj
      u_new = -μj*h .* G .+ νj .* u_cur .- (νj-1).*u_prev
      u_prev, u_cur = u_cur, u_new
    end
    u = copy(u_cur)
    t += h; iter += 1
    if !isnothing(maxit) && iter >= maxit
        break 
    end
  end

  val = cache.f(u, cache.p)[1]
  return SciMLBase.build_solution(cache, cache.opt, u, val;
    retcode = ReturnCode.Success,
    stats   = Optimization.OptimizationStats(iter, t, iter*s, iter*s, 0))
end

# Chebyshev polynomial helper
function chebyshevT(j::Integer, x)
  j==0 && return one(x)
  j==1 && return x
  T0, T1 = one(x), x
  for k in 2:j
    T0, T1 = T1, 2*x*T1 - T0
  end
  return T1
end

end
