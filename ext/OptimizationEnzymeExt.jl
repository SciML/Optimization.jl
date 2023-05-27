module OptimizationEnzymeExt

import SciMLBase: OptimizationFunction
import Optimization, ArrayInterface
import LinearAlgebra: I
import ADTypes: AutoEnzyme
isdefined(Base, :get_extension) ? (using Enzyme) : (using ..Enzyme)

function Optimization.instantiate_function(f::OptimizationFunction{true}, x,
                                           adtype::AutoEnzyme, p,
                                           num_cons = 0)
    _f = (θ, y, args...) -> (y .= first(f.f(θ, p, args...)); return nothing)

    if f.grad === nothing
        function grad(res, θ, args...)
            Enzyme.gradient!(Enzyme.Reverse, res, (θ) -> f.f(θ, p, args...), θ)
        end
    else
        grad = (G, θ, args...) -> f.grad(G, θ, p, args...)
    end

    if f.hess === nothing
        function hess(res, θ, args...)
            y = zeros(1)

            vdy = Tuple(zeros(1) for i in eachindex(θ))
            vdθ = Tuple((Array(r) for r in eachrow(I(length(θ)) * 1.0)))

            bθ = zeros(length(θ))
            by = ones(1)
            vdbθ = Tuple(zeros(length(θ)) for i in eachindex(θ))
            vdby = Tuple(zeros(1) for i in eachindex(θ))

            Enzyme.autodiff(Enzyme.Forward,
                            (θ, y) -> Enzyme.autodiff_deferred(Enzyme.Reverse, _f, θ, y),
                            Enzyme.BatchDuplicated(Enzyme.Duplicated(θ, bθ),
                                                   Enzyme.Duplicated.(vdθ, vdbθ)),
                            Enzyme.BatchDuplicated(Enzyme.Duplicated(y, by),
                                                   Enzyme.Duplicated.(vdy, vdby)))

            for i in eachindex(θ)
                res[i, :] .= vdbθ[i]
            end
        end
    else
        hess = (H, θ, args...) -> f.hess(H, θ, p, args...)
    end

    if f.hv === nothing
        hv = function (H, θ, v, args...)
            res = ArrayInterface.zeromatrix(θ)
            hess(res, θ, args...)
            H .= res * v
        end
    else
        hv = f.hv
    end

    if f.cons === nothing
        cons = nothing
    else
        cons = (res, θ) -> (f.cons(res, θ, p); return nothing)
        cons_oop = (x) -> (_res = zeros(eltype(x), num_cons); cons(_res, x); _res)
    end

    if cons !== nothing && f.cons_j === nothing
        cons_j = function (J, θ)
            if typeof(J) <: Vector
                J .= Enzyme.jacobian(Enzyme.Forward, cons_oop, θ)[1, :]
            else
                J .= Enzyme.jacobian(Enzyme.Forward, cons_oop, θ)
            end
        end
    else
        cons_j = (J, θ) -> f.cons_j(J, θ, p)
    end

    if cons !== nothing && f.cons_h === nothing
        fncs = map(1:num_cons) do i
            function (x)
                res = zeros(eltype(x), num_cons)
                f.cons(res, x, p)
                return res[i]
            end
        end
        function f2(fnc, x)
            dx = zeros(length(x))
            Enzyme.autodiff_deferred(Enzyme.Reverse, fnc, Enzyme.Duplicated(x, dx))
            dx
        end
        cons_h = function (res, θ)
            for i in 1:num_cons
                res[i] .= Enzyme.jacobian(Enzyme.Forward, x -> f2(fncs[i], x), θ)
            end
        end
    else
        cons_h = (res, θ) -> f.cons_h(res, θ, p)
    end

    return OptimizationFunction{true}(f.f, adtype; grad = grad, hess = hess, hv = hv,
                                      cons = cons, cons_j = cons_j, cons_h = cons_h,
                                      hess_prototype = f.hess_prototype,
                                      cons_jac_prototype = f.cons_jac_prototype,
                                      cons_hess_prototype = f.cons_hess_prototype)
end

end

function Optimization.instantiate_function(f::OptimizationFunction{true},
                                           cache::ReInitCache,
                                           adtype::AutoEnzyme,
                                           num_cons = 0)
    _f = (θ, y, args...) -> (y .= first(f.f(θ, cache.p, args...)); return nothing)

    if f.grad === nothing
        function grad(res, θ, args...)
            Enzyme.gradient!(Enzyme.Reverse, res, (θ) -> f.f(θ, cache.p, args...), θ)
        end
    else
        grad = (G, θ, args...) -> f.grad(G, θ, cache.p, args...)
    end

    if f.hess === nothing
        function hess(res, θ, args...)
            y = zeros(1)

            vdy = Tuple(zeros(1) for i in eachindex(θ))
            vdθ = Tuple((Array(r) for r in eachrow(I(length(θ)) * 1.0)))

            bθ = zeros(length(θ))
            by = ones(1)
            vdbθ = Tuple(zeros(length(θ)) for i in eachindex(θ))
            vdby = Tuple(zeros(1) for i in eachindex(θ))

            Enzyme.autodiff(Enzyme.Forward,
                            (θ, y) -> Enzyme.autodiff_deferred(Enzyme.Reverse, _f, θ, y),
                            Enzyme.BatchDuplicated(Enzyme.Duplicated(θ, bθ),
                                                   Enzyme.Duplicated.(vdθ, vdbθ)),
                            Enzyme.BatchDuplicated(Enzyme.Duplicated(y, by),
                                                   Enzyme.Duplicated.(vdy, vdby)))

            for i in eachindex(θ)
                res[i, :] .= vdbθ[i]
            end
        end
    else
        hess = (H, θ, args...) -> f.hess(H, θ, cache.p, args...)
    end

    if f.hv === nothing
        hv = function (H, θ, v, args...)
            res = ArrayInterface.zeromatrix(θ)
            hess(res, θ, args...)
            H .= res * v
        end
    else
        hv = f.hv
    end

    if f.cons === nothing
        cons = nothing
    else
        cons = (res, θ) -> (f.cons(res, θ, cache.p); return nothing)
        cons_oop = (x) -> (_res = zeros(eltype(x), num_cons); cons(_res, x); _res)
    end

    if cons !== nothing && f.cons_j === nothing
        cons_j = function (J, θ)
            if typeof(J) <: Vector
                J .= Enzyme.jacobian(Enzyme.Forward, cons_oop, θ)[1, :]
            else
                J .= Enzyme.jacobian(Enzyme.Forward, cons_oop, θ)
            end
        end
    else
        cons_j = (J, θ) -> f.cons_j(J, θ, cache.p)
    end

    if cons !== nothing && f.cons_h === nothing
        fncs = map(1:num_cons) do i
            function (x)
                res = zeros(eltype(x), num_cons)
                f.cons(res, x, cache.p)
                return res[i]
            end
        end
        function f2(fnc, x)
            dx = zeros(length(x))
            Enzyme.autodiff_deferred(Enzyme.Reverse, fnc, Enzyme.Duplicated(x, dx))
            dx
        end
        cons_h = function (res, θ)
            for i in 1:num_cons
                res[i] .= Enzyme.jacobian(Enzyme.Forward, x -> f2(fncs[i], x), θ)
            end
        end
    else
        cons_h = (res, θ) -> f.cons_h(res, θ, cache.p)
    end

    return OptimizationFunction{true}(f.f, adtype; grad = grad, hess = hess, hv = hv,
                                      cons = cons, cons_j = cons_j, cons_h = cons_h,
                                      hess_prototype = f.hess_prototype,
                                      cons_jac_prototype = f.cons_jac_prototype,
                                      cons_hess_prototype = f.cons_hess_prototype)
end
