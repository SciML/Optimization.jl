"""
AutoTracker <: AbstractADType

An AbstractADType choice for use in OptimizationFunction for automatically
generating the unspecified derivative functions. Usage:

```julia
OptimizationFunction(f,AutoTracker();kwargs...)
```

This uses the [Tracker.jl](https://github.com/FluxML/Tracker.jl) package.
Generally slower than ReverseDiff, it is generally applicable to many
pure Julia codes.

- Compatible with GPUs
- Not compatible with Hessian-based optimization
- Not compatible with Hv-based optimization
- Not compatible with constraint functions

Note that only the unspecified derivative functions are defined. For example,
if a `hess` function is supplied to the `OptimizationFunction`, then the
Hessian is not defined via Tracker.
"""
struct AutoTracker <: AbstractADType end

function instantiate_function(f, x, adtype::AutoTracker, p, num_cons = 0)
    num_cons != 0 && error("AutoTracker does not currently support constraints")
    _f = (θ, args...) -> first(f.f(θ, p, args...))

    if f.grad === nothing
        grad = (res, θ, args...) -> res isa DiffResults.DiffResult ?
                                    DiffResults.gradient!(res,
                                                          Tracker.data(Tracker.gradient(x -> _f(x,
                                                                                                args...),
                                                                                        θ)[1])) :
                                    res .= Tracker.data(Tracker.gradient(x -> _f(x, args...),
                                                                         θ)[1])
    else
        grad = f.grad
    end

    if f.hess === nothing
        hess = (res, θ, args...) -> error("Hessian based methods not supported with Tracker backend, pass in the `hess` kwarg")
    else
        hess = f.hess
    end

    if f.hv === nothing
        hv = (res, θ, args...) -> error("Hessian based methods not supported with Tracker backend, pass in the `hess` and `hv` kwargs")
    else
        hv = f.hv
    end

    return OptimizationFunction{false}(f, adtype; grad = grad, hess = hess, hv = hv,
                                       cons = nothing, cons_j = nothing, cons_h = nothing,
                                       hess_prototype = nothing,
                                       cons_jac_prototype = nothing,
                                       cons_hess_prototype = nothing)
end
