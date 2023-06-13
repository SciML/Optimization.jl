module OptimizationZygoteExt

import SciMLBase: OptimizationFunction
import Optimization
import ADTypes: AutoZygote
isdefined(Base, :get_extension) ? (using Zygote, Zygote.ForwardDiff) : (using ..Zygote, ..Zygote.ForwardDiff)

function Optimization.instantiate_function(f, x, adtype::AutoZygote, p,
                                           num_cons = 0)
    num_cons != 0 && error("AutoZygote does not currently support constraints")

    _f = (θ, args...) -> f(θ, p, args...)[1]
    if f.grad === nothing
        grad = (res, θ, args...) -> false ?
                                    false :
                                    res .= Zygote.gradient(x -> _f(x, args...), θ)[1]
    else
        grad = (G, θ, args...) -> f.grad(G, θ, p, args...)
    end

    if f.hess === nothing
        hess = function (res, θ, args...)
            res .= ForwardDiff.jacobian(θ) do θ
                Zygote.gradient(x -> _f(x, args...), θ)[1]
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
                                           adtype::AutoZygote, num_cons = 0)
    num_cons != 0 && error("AutoZygote does not currently support constraints")

    _f = (θ, args...) -> f(θ, cache.p, args...)[1]
    if f.grad === nothing
        grad = (res, θ, args...) -> false ?
                                    false :
                                    res .= Zygote.gradient(x -> _f(x, args...), θ)[1]
    else
        grad = (G, θ, args...) -> f.grad(G, θ, cache.p, args...)
    end

    if f.hess === nothing
        hess = function (res, θ, args...)
            res .= ForwardDiff.jacobian(θ) do θ
                Zygote.gradient(x -> _f(x, args...), θ)[1]
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
