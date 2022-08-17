"""
AutoReverseDiff <: AbstractADType

An AbstractADType choice for use in OptimizationFunction for automatically
generating the unspecified derivative functions. Usage:

```julia
OptimizationFunction(f,AutoReverseDiff();kwargs...)
```

This uses the [ReverseDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl)
package. `AutoReverseDiff` has a default argument, `compile`, which
denotes whether the reverse pass should be compiled. **`compile` should only
be set to `true` if `f` contains no branches (if statements, while loops)
otherwise it can produce incorrect derivatives!**.

`AutoReverseDiff` is generally applicable to many pure Julia codes,
and with `compile=true` it is one of the fastest options on code with
heavy scalar interactions. Hessian calculations are fast by mixing
ForwardDiff with ReverseDiff for forward-over-reverse. However, its
performance can falter when `compile=false`.

- Not compatible with GPUs
- Compatible with Hessian-based optimization by mixing with ForwardDiff
- Compatible with Hv-based optimization by mixing with ForwardDiff
- Not compatible with constraint functions

Note that only the unspecified derivative functions are defined. For example,
if a `hess` function is supplied to the `OptimizationFunction`, then the
Hessian is not defined via ReverseDiff.

## Constructor

```julia
AutoReverseDiff(;compile = false)
```

#### Note: currently compilation is not defined/used!
"""
struct AutoReverseDiff <: AbstractADType
    compile::Bool
end

AutoReverseDiff(; compile = false) = AutoReverseDiff(compile)

function instantiate_function(f, x, adtype::AutoReverseDiff, p = SciMLBase.NullParameters(),
                              num_cons = 0)
    num_cons != 0 && error("AutoReverseDiff does not currently support constraints")

    _f = (θ, args...) -> first(f.f(θ, p, args...))

    if f.grad === nothing
        grad = (res, θ, args...) -> ReverseDiff.gradient!(res, x -> _f(x, args...), θ,
                                                          ReverseDiff.GradientConfig(θ))
    else
        grad = f.grad
    end

    if f.hess === nothing
        hess = function (res, θ, args...)
            if res isa DiffResults.DiffResult
                DiffResults.hessian!(res,
                                     ForwardDiff.jacobian(θ) do θ
                                         ReverseDiff.gradient(x -> _f(x, args...), θ)[1]
                                     end)
            else
                res .= ForwardDiff.jacobian(θ) do θ
                    ReverseDiff.gradient(x -> _f(x, args...), θ)
                end
            end
        end
    else
        hess = f.hess
    end

    if f.hv === nothing
        hv = function (H, θ, v, args...)
            _θ = ForwardDiff.Dual.(θ, v)
            res = DiffResults.GradientResult(_θ)
            grad(res, _θ, args...)
            H .= getindex.(ForwardDiff.partials.(DiffResults.gradient(res)), 1)
        end
    else
        hv = f.hv
    end

    return OptimizationFunction{false}(f, adtype; grad = grad, hess = hess, hv = hv,
                                       cons = nothing, cons_j = nothing, cons_h = nothing,
                                       hess_prototype = nothing,
                                       cons_jac_prototype = nothing,
                                       cons_hess_prototype = nothing)
end
