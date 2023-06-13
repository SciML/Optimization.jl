module OptimizationTrackerExt

import SciMLBase: OptimizationFunction
import Optimization
import ADTypes: AutoTracker
isdefined(Base, :get_extension) ? (using Tracker) : (using ..Tracker)

function Optimization.instantiate_function(f, x, adtype::AutoTracker, p,
    num_cons = 0)
    num_cons != 0 && error("AutoTracker does not currently support constraints")
    _f = (θ, args...) -> first(f.f(θ, p, args...))

    if f.grad === nothing
        grad = (res, θ, args...) -> res .= Tracker.data(Tracker.gradient(x -> _f(x, args...),
            θ)[1])
    else
        grad = (G, θ, args...) -> f.grad(G, θ, p, args...)
    end

    if f.hess === nothing
        hess = (res, θ, args...) -> error("Hessian based methods not supported with Tracker backend, pass in the `hess` kwarg")
    else
        hess = (H, θ, args...) -> f.hess(H, θ, p, args...)
    end

    if f.hv === nothing
        hv = (res, θ, args...) -> error("Hessian based methods not supported with Tracker backend, pass in the `hess` and `hv` kwargs")
    else
        hv = f.hv
    end

    return OptimizationFunction{false}(f, adtype; grad = grad, hess = hess, hv = hv,
        cons = nothing, cons_j = nothing, cons_h = nothing,
        hess_prototype = f.hess_prototype,
        cons_jac_prototype = nothing,
        cons_hess_prototype = nothing)
end

function Optimization.instantiate_function(f, cache::Optimization.ReInitCache,
    adtype::AutoTracker, num_cons = 0)
    num_cons != 0 && error("AutoTracker does not currently support constraints")
    _f = (θ, args...) -> first(f.f(θ, cache.p, args...))

    if f.grad === nothing
        grad = (res, θ, args...) -> res .= Tracker.data(Tracker.gradient(x -> _f(x, args...),
            θ)[1])
    else
        grad = (G, θ, args...) -> f.grad(G, θ, cache.p, args...)
    end

    if f.hess === nothing
        hess = (res, θ, args...) -> error("Hessian based methods not supported with Tracker backend, pass in the `hess` kwarg")
    else
        hess = (H, θ, args...) -> f.hess(H, θ, cache.p, args...)
    end

    if f.hv === nothing
        hv = (res, θ, args...) -> error("Hessian based methods not supported with Tracker backend, pass in the `hess` and `hv` kwargs")
    else
        hv = f.hv
    end

    return OptimizationFunction{false}(f, adtype; grad = grad, hess = hess, hv = hv,
        cons = nothing, cons_j = nothing, cons_h = nothing,
        hess_prototype = f.hess_prototype,
        cons_jac_prototype = nothing,
        cons_hess_prototype = nothing)
end

end
