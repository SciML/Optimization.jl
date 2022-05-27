struct AutoTracker <: AbstractADType end

function instantiate_function(f, x, adtype::AutoTracker, p, num_cons = 0)
    num_cons != 0 && error("AutoTracker does not currently support constraints")
    _f = (θ, args...) -> first(f.f(θ, p, args...))

    if f.grad === nothing
        grad = (res, θ, args...) -> res isa DiffResults.DiffResult ? DiffResults.gradient!(res, Tracker.data(Tracker.gradient(x -> _f(x, args...), θ)[1])) : res .= Tracker.data(Tracker.gradient(x -> _f(x, args...), θ)[1])
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


    return OptimizationFunction{true}(f, adtype; grad=grad, hess=hess, hv=hv,
        cons=nothing, cons_j=nothing, cons_h=nothing,
        hess_prototype=nothing, cons_jac_prototype=nothing, cons_hess_prototype=nothing)
end
