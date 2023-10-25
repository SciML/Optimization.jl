module OptimizationZygoteExt

import Optimization
import Optimization.SciMLBase: OptimizationFunction
import Optimization.ADTypes: AutoZygote
isdefined(Base, :get_extension) ? (using Zygote, Zygote.ForwardDiff) :
(using ..Zygote, ..Zygote.ForwardDiff)

function Optimization.instantiate_function(f, x, adtype::AutoZygote, p,
    num_cons = 0)
    _f = (θ, args...) -> f(θ, p, args...)[1]
    if f.grad === nothing
        grad = function (res, θ, args...)
            val = Zygote.gradient(x -> _f(x, args...), θ)[1]
            if val === nothing
                res .= 0
            else
                res .= val
            end
        end
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

    if f.cons === nothing
        cons = nothing
    else
        cons = (res, θ) -> f.cons(res, θ, p)
        cons_oop = (x) -> (_res = Zygote.Buffer(x, num_cons); cons(_res, x); copy(_res))
    end

    if cons !== nothing && f.cons_j === nothing
        cons_j = function (J, θ)
            J .= first(Zygote.jacobian(cons_oop, θ))
        end
    else
        cons_j = (J, θ) -> f.cons_j(J, θ, p)
    end

    if cons !== nothing && f.cons_h === nothing
        fncs = [(x) -> cons_oop(x)[i] for i in 1:num_cons]
        cons_h = function (res, θ)
            for i in 1:num_cons
                res[i] .= Zygote.hessian(fncs[i], θ)
            end
        end
    else
        cons_h = (res, θ) -> f.cons_h(res, θ, p)
    end

    if f.lag_h === nothing
        lag_h = nothing # Consider implementing this
    else
        lag_h = (res, θ, σ, μ) -> f.lag_h(res, θ, σ, μ, p)
    end

    return OptimizationFunction{true}(f.f, adtype; grad = grad, hess = hess, hv = hv,
        cons = cons, cons_j = cons_j, cons_h = cons_h,
        hess_prototype = f.hess_prototype,
        cons_jac_prototype = f.cons_jac_prototype,
        cons_hess_prototype = f.cons_hess_prototype,
        lag_h, f.lag_hess_prototype)
end

function Optimization.instantiate_function(f, cache::Optimization.ReInitCache,
    adtype::AutoZygote, num_cons = 0)
    _f = (θ, args...) -> f(θ, cache.p, args...)[1]
    if f.grad === nothing
        grad = function (res, θ, args...)
            val = Zygote.gradient(x -> _f(x, args...), θ)[1]
            if val === nothing
                res .= 0
            else
                res .= val
            end
        end
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

    if f.cons === nothing
        cons = nothing
    else
        cons = (res, θ) -> f.cons(res, θ, cache.p)
        cons_oop = (x) -> (_res = Zygote.Buffer(x, num_cons); cons(_res, x); copy(_res))
    end

    if cons !== nothing && f.cons_j === nothing
        cons_j = function (J, θ)
            J .= first(Zygote.jacobian(cons_oop, θ))
        end
    else
        cons_j = (J, θ) -> f.cons_j(J, θ, cache.p)
    end

    if cons !== nothing && f.cons_h === nothing
        fncs = [(x) -> cons_oop(x)[i] for i in 1:num_cons]
        cons_h = function (res, θ)
            for i in 1:num_cons
                res[i] .= Zygote.hessian(fncs[i], θ)
            end
        end
    else
        cons_h = (res, θ) -> f.cons_h(res, θ, cache.p)
    end

    if f.lag_h === nothing
        lag_h = nothing # Consider implementing this
    else
        lag_h = (res, θ, σ, μ) -> f.lag_h(res, θ, σ, μ, cache.p)
    end

    return OptimizationFunction{true}(f.f, adtype; grad = grad, hess = hess, hv = hv,
        cons = cons, cons_j = cons_j, cons_h = cons_h,
        hess_prototype = f.hess_prototype,
        cons_jac_prototype = f.cons_jac_prototype,
        cons_hess_prototype = f.cons_hess_prototype,
        lag_h, f.lag_hess_prototype)
end

end
