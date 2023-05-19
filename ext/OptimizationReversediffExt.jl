module OptimizationReversediffExt

import SciMLBase: OptimizationFunction, AbstractADType
import Optimization
import ADTypes: AutoReverseDiff
isdefined(Base, :get_extension) ? (using ReverseDiff) : (using ..ReverseDiff)
import ReverseDiff.ForwardDiff

function Optimization.instantiate_function(f, x, adtype::AutoReverseDiff,
                                           p = SciMLBase.NullParameters(),
                                           num_cons = 0)
    num_cons != 0 && error("AutoReverseDiff does not currently support constraints")

    _f = (θ, args...) -> first(f.f(θ, p, args...))

    if f.grad === nothing
        grad = (res, θ, args...) -> ReverseDiff.gradient!(res, x -> _f(x, args...), θ,
                                                          ReverseDiff.GradientConfig(θ))
    else
        grad = (G, θ, args...) -> f.grad(G, θ, p, args...)
    end

    if f.hess === nothing
        hess = function (res, θ, args...)
            res .= ForwardDiff.jacobian(θ) do θ
                ReverseDiff.gradient(x -> _f(x, args...), θ)
            end
        end
    else
        hess = (H, θ, args...) -> f.hess(H, θ, p, args...)
    end

    if f.hv === nothing
        hv = function (H, θ, v, args...)
            _θ = ForwardDiff.Dual.(θ, v)
            res = similar(_θ)
            grad(res, _θ, args...)
            H .= getindex.(ForwardDiff.partials.(res), 1)
        end
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
                                           adtype::AutoReverseDiff, num_cons = 0)
    num_cons != 0 && error("AutoReverseDiff does not currently support constraints")

    _f = (θ, args...) -> first(f.f(θ, cache.p, args...))

    if f.grad === nothing
        grad = (res, θ, args...) -> ReverseDiff.gradient!(res, x -> _f(x, args...), θ,
                                                          ReverseDiff.GradientConfig(θ))
    else
        grad = (G, θ, args...) -> f.grad(G, θ, cache.p, args...)
    end

    if f.hess === nothing
        hess = function (res, θ, args...)
            res .= ForwardDiff.jacobian(θ) do θ
                ReverseDiff.gradient(x -> _f(x, args...), θ)
            end
        end
    else
        hess = (H, θ, args...) -> f.hess(H, θ, cache.p, args...)
    end

    if f.hv === nothing
        hv = function (H, θ, v, args...)
            _θ = ForwardDiff.Dual.(θ, v)
            res = similar(_θ)
            grad(res, _θ, args...)
            H .= getindex.(ForwardDiff.partials.(res), 1)
        end
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
