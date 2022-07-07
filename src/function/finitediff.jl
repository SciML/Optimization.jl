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
- `fdjtype`: the method used for defining the Jacobian
- `fdhtype`: the method used for defining the Hessian

For more information on the derivative type specifiers, see the
[FiniteDiff.jl documentation](https://github.com/JuliaDiff/FiniteDiff.jl).
"""
struct AutoFiniteDiff{T1,T2,T3} <: AbstractADType
    fdtype::T1
    fdjtype::T2
    fdhtype::T3
end

AutoFiniteDiff(; fdtype=Val(:forward), fdjtype=fdtype, fdhtype=Val(:hcentral)) =
    AutoFiniteDiff(fdtype, fdjtype, fdhtype)

function instantiate_function(f, x, adtype::AutoFiniteDiff, p, num_cons=0)
    _f = (θ, args...) -> first(f.f(θ, p, args...))

    if f.grad === nothing
        grad = (res, θ, args...) -> FiniteDiff.finite_difference_gradient!(res, x -> _f(x, args...), θ, FiniteDiff.GradientCache(res, x, adtype.fdtype))
    else
        grad = f.grad
    end

    if f.hess === nothing
        hess = (res, θ, args...) -> FiniteDiff.finite_difference_hessian!(res, x -> _f(x, args...), θ, FiniteDiff.HessianCache(x, adtype.fdhtype))
    else
        hess = f.hess
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
        cons = θ -> f.cons(θ, p)
    end

    if cons !== nothing && f.cons_j === nothing
        function iip_cons(dx, x) # need f(dx, x) for jacobian? 
            dx .= [cons(x)[i] for i in 1:num_cons] # Not very efficient probably?
            nothing
        end
        cons_j = function (J, θ)
            y0 = zeros(num_cons)
            FiniteDiff.finite_difference_jacobian!(J, iip_cons, θ, FiniteDiff.JacobianCache(copy(θ), copy(y0), copy(y0), adtype.fdjtype))
        end
    else
        cons_j = f.cons_j
    end

    if cons !== nothing && f.cons_h === nothing
        cons_h = function (res, θ)
            for i in 1:num_cons
                FiniteDiff.finite_difference_hessian!(res[i], (x) -> cons(x)[i], θ, FiniteDiff.HessianCache(copy(θ), adtype.fdhtype))
            end
        end
    else
        cons_h = f.cons_h
    end

    return OptimizationFunction{true}(f, adtype; grad=grad, hess=hess, hv=hv,
        cons=cons, cons_j=cons_j, cons_h=cons_h,
        hess_prototype=nothing, cons_jac_prototype=nothing, cons_hess_prototype=nothing)
end
