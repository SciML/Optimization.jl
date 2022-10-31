"""
AutoFiniteDiff{T1,T2,T3} <: AbstractADType

An AbstractADType choice for use in OptimizationFunction for automatically
generating the unspecified derivative functions. Usage:

```julia
OptimizationFunction(f,AutoFiniteDiff();kwargs...)
```

This uses [FiniteDiff.jl](https://github.com/JuliaDiff/FiniteDiff.jl).
While to necessarily the most efficient in any case, this is the only
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
AutoFiniteDiff(;fdtype = Val(:forward) fdjtype = fdtype, fdhtype = Val(:hcentral))
```

- `fdtype`: the method used for defining the gradient
- `fdjtype`: the method used for defining the Jacobian of constraints.
- `fdhtype`: the method used for defining the Hessian
- `relstep`: the relative step size in finite differencing
- `absstep`: the absolute step size in finite differencing

For more information on the derivative type specifiers and step sizes, see the
[FiniteDiff.jl documentation](https://github.com/JuliaDiff/FiniteDiff.jl).
"""
struct AutoFiniteDiff{T1, T2, T3} <: AbstractADType
    fdtype::T1
    fdjtype::T2
    fdhtype::T3
    relstep::Float64
    absstep::Float64
end

function AutoFiniteDiff(; fdtype = Val(:forward), fdjtype = fdtype,
                        fdhtype = Val(:hcentral), relstep=FiniteDiff.default_relstep(fdtype, Float64), absstep=relstep)
    AutoFiniteDiff(fdtype, fdjtype, fdhtype, relstep, absstep)
end

function instantiate_function(f, x, adtype::AutoFiniteDiff, p, num_cons = 0)
    _f = (θ, args...) -> first(f.f(θ, p, args...))
    updatecache = (cache, x) -> (cache.xmm .= x; cache.xmp .= x; cache.xpm .= x; cache.xpp .= x; return cache)

    if f.grad === nothing
        gradcache = FiniteDiff.GradientCache(x, x, adtype.fdtype)
        grad = (res, θ, args...) -> FiniteDiff.finite_difference_gradient!(res,
                                                                           x -> _f(x,
                                                                                   args...),
                                                                           θ, gradcache, relstep = adtype.relstep, absstep = adtype.absstep)
    else
        grad = (G, θ, args...) -> f.grad(G, θ, p, args...)
    end

    if f.hess === nothing
        hesscache = FiniteDiff.HessianCache(x, adtype.fdhtype)
        hess = (res, θ, args...) -> FiniteDiff.finite_difference_hessian!(res,
                                                                          x -> _f(x,
                                                                                  args...),
                                                                          θ,
                                                                          updatecache(hesscache,
                                                                                      θ), relstep = adtype.relstep, absstep = adtype.absstep)
    else
        hess = (H, θ, args...) -> f.hess(H, θ, p, args...)
    end

    if f.hv === nothing
        hv = function (H, θ, v, args...)
            res = ArrayInterfaceCore.zeromatrix(θ)
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
            jaccache = FiniteDiff.JacobianCache(copy(x), copy(y0), copy(y0), adtype.fdjtype;
                                                colorvec = cons_jac_colorvec,
                                                sparsity = f.cons_jac_prototype)
            FiniteDiff.finite_difference_jacobian!(J, cons, θ, jaccache, relstep = adtype.relstep, absstep = adtype.absstep)
        end
    else
        cons_j = (J, θ) -> f.cons_j(J, θ, p)
    end

    if cons !== nothing && f.cons_h === nothing
        hess_cons_cache = [FiniteDiff.HessianCache(copy(x), adtype.fdhtype)
                           for i in 1:num_cons]
        cons_h = function (res, θ)
            for i in 1:num_cons#note: colorvecs not yet supported by FiniteDiff for Hessians
                FiniteDiff.finite_difference_hessian!(res[i],
                                                      (x) -> (_res = zeros(eltype(θ),
                                                                           num_cons);
                                                              cons(_res,
                                                                   x);
                                                              _res[i]),
                                                      θ, updatecache(hess_cons_cache[i], θ), relstep = adtype.relstep, absstep = adtype.absstep)
            end
        end
    else
        cons_h = (res, θ) -> f.cons_h(res, θ, p)
    end

    return OptimizationFunction{true}(f, adtype; grad = grad, hess = hess, hv = hv,
                                      cons = cons, cons_j = cons_j, cons_h = cons_h,
                                      cons_jac_colorvec = cons_jac_colorvec,
                                      hess_prototype = f.hess_prototype,
                                      cons_jac_prototype = f.cons_jac_prototype,
                                      cons_hess_prototype = f.cons_hess_prototype)
end
