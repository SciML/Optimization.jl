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
            dθ = zero(res)
            Enzyme.autodiff(Enzyme.Reverse, _f, Enzyme.Duplicated(θ, dθ),
                            Enzyme.DuplicatedNoNeed(zeros(1), ones(1)), args...)
            res .= dθ
        end
    else
        grad = (G, θ, args...) -> f.grad(G, θ, p, args...)
    end

    if f.hess === nothing
        function g(θ, bθ, y, by, args...)
            Enzyme.autodiff_deferred(Enzyme.Reverse, _f, Enzyme.Duplicated(θ, bθ), Enzyme.DuplicatedNoNeed(y, by), args...)
            return nothing
        end
        function hess(res, θ, args...)
            y = Vector{Float64}(undef, 1)

            vdθ = Tuple((Array(r) for r in eachrow(I(length(θ)) * 1.0)))

            bθ = zeros(length(θ))
            by = ones(1)
            vdbθ = Tuple(zeros(length(θ)) for i in eachindex(θ))

            Enzyme.autodiff(Enzyme.Forward,
                g,
                Enzyme.BatchDuplicated(θ, vdθ),
                Enzyme.BatchDuplicated(bθ, vdbθ),
                Const(y),
                Const(by),
                args...)

            for i in eachindex(θ)
                res[i, :] .= vdbθ[i]
            end
        end
    else
        hess = (H, θ, args...) -> f.hess(H, θ, p, args...)
    end

    if f.hv === nothing
        hv = function (H, θ, v, args...)
            function f2(x, v, args...)::Float64
                dx = zeros(length(x))
                Enzyme.autodiff_deferred(Enzyme.Reverse, _f,
                                         Enzyme.Duplicated(x, dx),
                                         Enzyme.DuplicatedNoNeed(zeros(1), ones(1)),
                                         args...)
                Float64(dot(dx, v))
            end
            H .= Enzyme.gradient(Enzyme.Forward, x -> f2(x, v, args...), θ)
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

function Optimization.instantiate_function(f::OptimizationFunction{true},
    cache::Optimization.ReInitCache,
    adtype::AutoEnzyme,
    num_cons = 0)
    _f = (θ, y, args...) -> (y .= first(f.f(θ, cache.p, args...)); return nothing)

    if f.grad === nothing
        function grad(res, θ, args...)
            dθ = zero(res)
            Enzyme.autodiff(Enzyme.Reverse, _f, Enzyme.Duplicated(θ, dθ),
                            Enzyme.DuplicatedNoNeed(zeros(1), ones(1)), args...)
            res .= dθ
        end
    else
        grad = (G, θ, args...) -> f.grad(G, θ, p, args...)
    end

    if f.hess === nothing
        function g(θ, bθ, y, by, args...)
            Enzyme.autodiff_deferred(Enzyme.Reverse, _f, Enzyme.Duplicated(θ, bθ),
                                     Enzyme.DuplicatedNoNeed(y, by), args...)
            return nothing
        end
        function hess(res, θ, args...)
            y = Vector{Float64}(undef, 1)

            vdθ = Tuple((Array(r) for r in eachrow(I(length(θ)) * 1.0)))

            bθ = zeros(length(θ))
            by = ones(1)
            vdbθ = Tuple(zeros(length(θ)) for i in eachindex(θ))

            Enzyme.autodiff(Enzyme.Forward,
                            g,
                            Enzyme.BatchDuplicated(θ, vdθ),
                            Enzyme.BatchDuplicated(bθ, vdbθ),
                            Const(y),
                            Const(by),
                            args...)

            for i in eachindex(θ)
                res[i, :] .= vdbθ[i]
            end
        end
    else
        hess = (H, θ, args...) -> f.hess(H, θ, p, args...)
    end

    if f.hv === nothing
        hv = function (H, θ, v, args...)
            function f2(x, v, args...)::Float64
                dx = zeros(length(x))
                Enzyme.autodiff_deferred(Enzyme.Reverse, _f,
                                         Enzyme.Duplicated(x, dx),
                                         Enzyme.DuplicatedNoNeed(zeros(1), ones(1)),
                                         args...)
                Float64(dot(dx, v))
            end
            H .= Enzyme.gradient(Enzyme.Forward, x -> f2(x, v, args...), θ)
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

end
