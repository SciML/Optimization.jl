"""
AutoZygote <: AbstractADType

An AbstractADType choice for use in OptimizationFunction for automatically
generating the unspecified derivative functions. Usage:

```julia
OptimizationFunction(f,AutoZygote();kwargs...)
```

This uses the [Zygote.jl](https://github.com/FluxML/Zygote.jl) package.
This is the staple reverse-mode AD that handles a large portion of
Julia with good efficiency. Hessian construction is fast via
forward-over-reverse mixing ForwardDiff.jl with Zygote.jl

- Compatible with GPUs
- Compatible with Hessian-based optimization via ForwardDiff
- Compatible with Hv-based optimization via ForwardDiff
- Not compatible with constraint functions

Note that only the unspecified derivative functions are defined. For example,
if a `hess` function is supplied to the `OptimizationFunction`, then the
Hessian is not defined via Zygote.
"""
struct AutoZygote <: AbstractADType end

function instantiate_function(f, x, adtype::AutoZygote, p, num_cons = 0)
    num_cons != 0 && error("AutoZygote does not currently support constraints")

    _f = (θ, args...) -> f(θ, p, args...)[1]
    if f.grad === nothing
        grad = (res, θ, args...) -> res isa DiffResults.DiffResult ?
                                    DiffResults.gradient!(res,
                                                          Zygote.gradient(x -> _f(x,
                                                                                  args...),
                                                                          θ)[1]) :
                                    res .= Zygote.gradient(x -> _f(x, args...), θ)[1]
    else
        grad = f.grad
    end

    if f.hess === nothing
        hess = function (res, θ, args...)
            if res isa DiffResults.DiffResult
                DiffResults.hessian!(res, ForwardDiff.jacobian(θ) do θ
                                         Zygote.gradient(x -> _f(x, args...), θ)[1]
                                     end)
            else
                res .= ForwardDiff.jacobian(θ) do θ
                    Zygote.gradient(x -> _f(x, args...), θ)[1]
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
