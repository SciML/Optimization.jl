module OptimizationFinitediffExt

import SciMLBase: OptimizationFunction, AbstractADType
import Optimization
isdefined(Base, :get_extension) ? (using FiniteDiff) : (using ..FiniteDiff)

const FD = FiniteDiff
"""
AutoFiniteDiff{T1,T2,T3} <: AbstractADType

An AbstractADType choice for use in OptimizationFunction for automatically
generating the unspecified derivative functions. Usage:

```julia
OptimizationFunction(f, AutoFiniteDiff(); kwargs...)
```

This uses [FiniteDiff.jl](https://github.com/JuliaDiff/FiniteDiff.jl).
While not necessarily the most efficient, this is the only
choice that doesn't require the `f` function to be automatically
differentiable, which means it applies to any choice. However, because
it's using finite differencing, one needs to be careful as this procedure
introduces numerical error into the derivative estimates.

  - Compatible with GPUs
  - Compatible with Hessian-based optimization
  - Compatible with Hv-based optimization
  - Compatible with constraint functions

Note that only the unspecified derivative functions are defined. For example,
if a `hess` function is supplied to the `OptimizationFunction`, then the
Hessian is not defined via FiniteDiff.

## Constructor

```julia
AutoFiniteDiff(; fdtype = Val(:forward)fdjtype = fdtype, fdhtype = Val(:hcentral))
```

  - `fdtype`: the method used for defining the gradient
  - `fdjtype`: the method used for defining the Jacobian of constraints.
  - `fdhtype`: the method used for defining the Hessian

For more information on the derivative type specifiers, see the
[FiniteDiff.jl documentation](https://github.com/JuliaDiff/FiniteDiff.jl).
"""
struct AutoFiniteDiff{T1, T2, T3} <: AbstractADType
    fdtype::T1
    fdjtype::T2
    fdhtype::T3
end

function AutoFiniteDiff(; fdtype = Val(:forward), fdjtype = fdtype,
                        fdhtype = Val(:hcentral))
    AutoFiniteDiff(fdtype, fdjtype, fdhtype)
end

function Optimization.instantiate_function(f, x, adtype::AutoFiniteDiff, p,
                                           num_cons = 0)
    _f = (θ, args...) -> first(f.f(θ, p, args...))
    updatecache = (cache, x) -> (cache.xmm .= x; cache.xmp .= x; cache.xpm .= x; cache.xpp .= x; return cache)

    if f.grad === nothing
        gradcache = FD.GradientCache(x, x, adtype.fdtype)
        grad = (res, θ, args...) -> FD.finite_difference_gradient!(res, x -> _f(x, args...),
                                                                   θ, gradcache)
    else
        grad = (G, θ, args...) -> f.grad(G, θ, p, args...)
    end

    if f.hess === nothing
        hesscache = FD.HessianCache(x, adtype.fdhtype)
        hess = (res, θ, args...) -> FD.finite_difference_hessian!(res,
                                                                  x -> _f(x, args...), θ,
                                                                  updatecache(hesscache, θ))
    else
        hess = (H, θ, args...) -> f.hess(H, θ, p, args...)
    end

    if f.hv === nothing
        hv = function (H, θ, v, args...)
            res = ArrayInterface.zeromatrix(θ)
            hess(res, θ, args...)
            H .= res * v
        end
    else
        hv = f.hv
    end

    if f.cons === nothing
        cons = nothing
    else
        cons = (res, θ) -> f.cons(res, θ, p)
    end

    cons_jac_colorvec = f.cons_jac_colorvec === nothing ? (1:length(x)) :
                        f.cons_jac_colorvec

    if cons !== nothing && f.cons_j === nothing
        cons_j = function (J, θ)
            y0 = zeros(num_cons)
            jaccache = FD.JacobianCache(copy(x), copy(y0), copy(y0), adtype.fdjtype;
                                        colorvec = cons_jac_colorvec,
                                        sparsity = f.cons_jac_prototype)
            FD.finite_difference_jacobian!(J, cons, θ, jaccache)
        end
    else
        cons_j = (J, θ) -> f.cons_j(J, θ, p)
    end

    if cons !== nothing && f.cons_h === nothing
        hess_cons_cache = [FD.HessianCache(copy(x), adtype.fdhtype)
                           for i in 1:num_cons]
        cons_h = function (res, θ)
            for i in 1:num_cons#note: colorvecs not yet supported by FiniteDiff for Hessians
                FD.finite_difference_hessian!(res[i],
                                              (x) -> (_res = zeros(eltype(θ), num_cons);
                                                      cons(_res, x);
                                                      _res[i]), θ,
                                              updatecache(hess_cons_cache[i], θ))
            end
        end
    else
        cons_h = (res, θ) -> f.cons_h(res, θ, p)
    end

    if f.lag_h === nothing
        lag_hess_cache = FD.HessianCache(copy(x), adtype.fdhtype)
        c = zeros(num_cons)
        h = zeros(length(x), length(x))
        lag_h = let c = c, h = h
            lag = function (θ, σ, μ)
                f.cons(c, θ, p)
                l = μ'c
                if !iszero(σ)
                    l += σ * f.f(θ, p)
                end
                l
            end
            function (res, θ, σ, μ)
                FD.finite_difference_hessian!(res,
                                              (x) -> lag(x, σ, μ),
                                              θ,
                                              updatecache(lag_hess_cache, θ))
            end
        end
    else
        lag_h = (res, θ, σ, μ) -> f.lag_h(res, θ, σ, μ, p)
    end
    return OptimizationFunction{true}(f, adtype; grad = grad, hess = hess, hv = hv,
                                      cons = cons, cons_j = cons_j, cons_h = cons_h,
                                      cons_jac_colorvec = cons_jac_colorvec,
                                      hess_prototype = f.hess_prototype,
                                      cons_jac_prototype = f.cons_jac_prototype,
                                      cons_hess_prototype = f.cons_hess_prototype,
                                      lag_h, f.lag_hess_prototype)
end

function Optimization.instantiate_function(f, cache::Optimization.ReInitCache,
                                           adtype::AutoFiniteDiff, num_cons = 0)
    _f = (θ, args...) -> first(f.f(θ, cache.p, args...))
    updatecache = (cache, x) -> (cache.xmm .= x; cache.xmp .= x; cache.xpm .= x; cache.xpp .= x; return cache)

    if f.grad === nothing
        gradcache = FD.GradientCache(cache.u0, cache.u0, adtype.fdtype)
        grad = (res, θ, args...) -> FD.finite_difference_gradient!(res, x -> _f(x, args...),
                                                                   θ, gradcache)
    else
        grad = (G, θ, args...) -> f.grad(G, θ, cache.p, args...)
    end

    if f.hess === nothing
        hesscache = FD.HessianCache(cache.u0, adtype.fdhtype)
        hess = (res, θ, args...) -> FD.finite_difference_hessian!(res, x -> _f(x, args...),
                                                                  θ,
                                                                  updatecache(hesscache, θ))
    else
        hess = (H, θ, args...) -> f.hess(H, θ, cache.p, args...)
    end

    if f.hv === nothing
        hv = function (H, θ, v, args...)
            res = ArrayInterface.zeromatrix(θ)
            hess(res, θ, args...)
            H .= res * v
        end
    else
        hv = f.hv
    end

    if f.cons === nothing
        cons = nothing
    else
        cons = (res, θ) -> f.cons(res, θ, cache.p)
    end

    cons_jac_colorvec = f.cons_jac_colorvec === nothing ? (1:length(cache.u0)) :
                        f.cons_jac_colorvec

    if cons !== nothing && f.cons_j === nothing
        cons_j = function (J, θ)
            y0 = zeros(num_cons)
            jaccache = FD.JacobianCache(copy(cache.u0), copy(y0), copy(y0),
                                        adtype.fdjtype;
                                        colorvec = cons_jac_colorvec,
                                        sparsity = f.cons_jac_prototype)
            FD.finite_difference_jacobian!(J, cons, θ, jaccache)
        end
    else
        cons_j = (J, θ) -> f.cons_j(J, θ, cache.p)
    end

    if cons !== nothing && f.cons_h === nothing
        hess_cons_cache = [FD.HessianCache(copy(cache.u0), adtype.fdhtype)
                           for i in 1:num_cons]
        cons_h = function (res, θ)
            for i in 1:num_cons#note: colorvecs not yet supported by FiniteDiff for Hessians
                FD.finite_difference_hessian!(res[i],
                                              (x) -> (_res = zeros(eltype(θ), num_cons);
                                                      cons(_res,
                                                           x);
                                                      _res[i]),
                                              θ, updatecache(hess_cons_cache[i], θ))
            end
        end
    else
        cons_h = (res, θ) -> f.cons_h(res, θ, cache.p)
    end
    if f.lag_h === nothing
        lag_hess_cache = FD.HessianCache(copy(cache.u0), adtype.fdhtype)
        c = zeros(num_cons)
        h = zeros(length(cache.u0), length(cache.u0))
        lag_h = let c = c, h = h
            lag = function (θ, σ, μ)
                f.cons(c, θ, cache.p)
                l = μ'c
                if !iszero(σ)
                    l += σ * f.f(θ, cache.p)
                end
                l
            end
            function (res, θ, σ, μ)
                FD.finite_difference_hessian!(h,
                                              (x) -> lag(x, σ, μ),
                                              θ,
                                              updatecache(lag_hess_cache, θ))
                k = 1
                for i in 1:length(cache.u0), j in i:length(cache.u0)
                    res[k] = h[i, j]
                    k += 1
                end
            end
        end
    else
        lag_h = (res, θ, σ, μ) -> f.lag_h(res, θ, σ, μ, cache.p)
    end
    return OptimizationFunction{true}(f, adtype; grad = grad, hess = hess, hv = hv,
                                      cons = cons, cons_j = cons_j, cons_h = cons_h,
                                      cons_jac_colorvec = cons_jac_colorvec,
                                      hess_prototype = f.hess_prototype,
                                      cons_jac_prototype = f.cons_jac_prototype,
                                      cons_hess_prototype = f.cons_hess_prototype,
                                      lag_h, f.lag_hess_prototype)
end

export AutoFiniteDiff
end
