module OptimizationSparseFiniteDiffExt

import Optimization, Optimization.ArrayInterface
import Optimization.SciMLBase: OptimizationFunction
import Optimization.ADTypes: AutoSparseFiniteDiff
using Optimization.LinearAlgebra
isdefined(Base, :get_extension) ? (using FiniteDiff, SparseDiffTools, Symbolics) :
(using ..FiniteDiff, ..SparseDiffTools, ..Symbolics)

const FD = FiniteDiff

function Optimization.instantiate_function(f, x, adtype::AutoSparseFiniteDiff, p,
    num_cons = 0)
    if maximum(getfield.(methods(f.f), :nargs)) > 3
        error("$(string(adtype)) with SparseDiffTools does not support functions with more than 2 arguments")
    end

    _f = (θ, args...) -> first(f.f(θ, p, args...))

    if f.grad === nothing
        gradcache = FD.GradientCache(x, x)
        grad = (res, θ, args...) -> FD.finite_difference_gradient!(res, x -> _f(x, args...),
            θ, gradcache)
    else
        grad = (G, θ, args...) -> f.grad(G, θ, p, args...)
    end

    if f.hess === nothing
        hess_sparsity = Symbolics.hessian_sparsity(_f, x)
        hess_colors = matrix_colors(tril(hess_sparsity))
        hess = (res, θ, args...) -> numauto_color_hessian!(res, x -> _f(x, args...), θ,
            ForwardColorHesCache(_f, x,
                hess_colors,
                hess_sparsity,
                (res, θ) -> grad(res,
                    θ,
                    args...)))
    else
        hess = (H, θ, args...) -> f.hess(H, θ, p, args...)
    end

    if f.hv === nothing
        hv = function (H, θ, v, args...)
            num_hesvec!(H, x -> _f(x, args...), θ, v)
        end
    else
        hv = f.hv
    end

    if f.cons === nothing
        cons = nothing
    else
        cons = (res, θ) -> f.cons(res, θ, p)
    end

    if cons !== nothing && f.cons_j === nothing
        cons_jac_prototype = f.cons_jac_prototype === nothing ?
                             Symbolics.jacobian_sparsity(cons,
            zeros(eltype(x), num_cons),
            x) :
                             f.cons_jac_prototype
        cons_jac_colorvec = f.cons_jac_colorvec === nothing ?
                            matrix_colors(tril(cons_jac_prototype)) :
                            f.cons_jac_colorvec
        cons_j = function (J, θ)
            y0 = zeros(num_cons)
            jaccache = FD.JacobianCache(copy(x), copy(y0), copy(y0);
                colorvec = cons_jac_colorvec,
                sparsity = cons_jac_prototype)
            FD.finite_difference_jacobian!(J, cons, θ, jaccache)
        end
    else
        cons_j = (J, θ) -> f.cons_j(J, θ, p)
    end

    if cons !== nothing && f.cons_h === nothing
        function gen_conshess_cache(_f, x)
            conshess_sparsity = Symbolics.hessian_sparsity(_f, x)
            conshess_colors = matrix_colors(conshess_sparsity)
            hesscache = ForwardColorHesCache(_f, x, conshess_colors, conshess_sparsity)
            return hesscache
        end

        fcons = [(x) -> (_res = zeros(eltype(x), num_cons);
        cons(_res, x);
        _res[i]) for i in 1:num_cons]

        cons_h = function (res, θ)
            for i in 1:num_cons
                numauto_color_hessian!(res[i], fcons[i], θ, gen_conshess_cache(fcons[i], θ))
            end
        end
    else
        cons_h = (res, θ) -> f.cons_h(res, θ, p)
    end

    if f.lag_h === nothing
        lag_hess_cache = FD.HessianCache(copy(x))
        c = zeros(num_cons)
        h = zeros(length(x), length(x))
        lag_h = let c = c, h = h
            lag = function (θ, σ, μ)
                f.cons(c, θ, p)
                l = μ'c
                if !iszero(σ)
                    l += σ * f.f(θ, p)
                end
                l
            end
            function (res, θ, σ, μ)
                FD.finite_difference_hessian!(res,
                    (x) -> lag(x, σ, μ),
                    θ,
                    updatecache(lag_hess_cache, θ))
            end
        end
    else
        lag_h = (res, θ, σ, μ) -> f.lag_h(res, θ, σ, μ, p)
    end
    return OptimizationFunction{true}(f, adtype; grad = grad, hess = hess, hv = hv,
        cons = cons, cons_j = cons_j, cons_h = cons_h,
        cons_jac_colorvec = f.cons_jac_colorvec,
        hess_prototype = f.hess_prototype,
        cons_jac_prototype = f.cons_jac_prototype,
        cons_hess_prototype = f.cons_hess_prototype,
        lag_h, f.lag_hess_prototype)
end

function Optimization.instantiate_function(f, cache::Optimization.ReInitCache,
    adtype::AutoSparseFiniteDiff, num_cons = 0)
    if maximum(getfield.(methods(f.f), :nargs)) > 3
        error("$(string(adtype)) with SparseDiffTools does not support functions with more than 2 arguments")
    end
    _f = (θ, args...) -> first(f.f(θ, cache.p, args...))
    updatecache = (cache, x) -> (cache.xmm .= x; cache.xmp .= x; cache.xpm .= x; cache.xpp .= x; return cache)

    if f.grad === nothing
        gradcache = FD.GradientCache(cache.u0, cache.u0)
        grad = (res, θ, args...) -> FD.finite_difference_gradient!(res, x -> _f(x, args...),
            θ, gradcache)
    else
        grad = (G, θ, args...) -> f.grad(G, θ, cache.p, args...)
    end

    if f.hess === nothing
        hess_sparsity = Symbolics.hessian_sparsity(_f, cache.u0)
        hess_colors = matrix_colors(tril(hess_sparsity))
        hess = (res, θ, args...) -> numauto_color_hessian!(res, x -> _f(x, args...), θ,
            ForwardColorHesCache(_f, θ,
                hess_colors,
                hess_sparsity,
                (res, θ) -> grad(res,
                    θ,
                    args...)))
    else
        hess = (H, θ, args...) -> f.hess(H, θ, cache.p, args...)
    end

    if f.hv === nothing
        hv = function (H, θ, v, args...)
            num_hesvec!(H, x -> _f(x, args...), θ, v)
        end
    else
        hv = f.hv
    end

    if f.cons === nothing
        cons = nothing
    else
        cons = (res, θ) -> f.cons(res, θ, cache.p)
    end

    if cons !== nothing && f.cons_j === nothing
        cons_jac_prototype = f.cons_jac_prototype === nothing ?
                             Symbolics.jacobian_sparsity(cons, zeros(eltype(x), num_cons),
            x) :
                             f.cons_jac_prototype
        cons_jac_colorvec = f.cons_jac_colorvec === nothing ?
                            matrix_colors(tril(cons_jac_prototype)) :
                            f.cons_jac_colorvec
        cons_j = function (J, θ)
            y0 = zeros(num_cons)
            jaccache = FD.JacobianCache(copy(x), copy(y0), copy(y0);
                colorvec = cons_jac_colorvec,
                sparsity = cons_jac_prototype)
            FD.finite_difference_jacobian!(J, cons, θ, jaccache)
        end
    else
        cons_j = (J, θ) -> f.cons_j(J, θ, cache.p)
    end

    if cons !== nothing && f.cons_h === nothing
        function gen_conshess_cache(_f, x)
            conshess_sparsity = copy(Symbolics.hessian_sparsity(_f, x))
            conshess_colors = matrix_colors(conshess_sparsity)
            hesscache = ForwardColorHesCache(_f, x, conshess_colors,
                conshess_sparsity)
            return hesscache
        end

        fcons = [(x) -> (_res = zeros(eltype(x), num_cons);
        cons(_res, x);
        _res[i]) for i in 1:num_cons]
        cons_h = function (res, θ)
            for i in 1:num_cons
                numauto_color_hessian!(res[i], fcons[i], θ, gen_conshess_cache(fcons[i], θ))
            end
        end
    else
        cons_h = (res, θ) -> f.cons_h(res, θ, cache.p)
    end
    if f.lag_h === nothing
        lag_hess_cache = FD.HessianCache(copy(cache.u0))
        c = zeros(num_cons)
        h = zeros(length(cache.u0), length(cache.u0))
        lag_h = let c = c, h = h
            lag = function (θ, σ, μ)
                f.cons(c, θ, cache.p)
                l = μ'c
                if !iszero(σ)
                    l += σ * f.f(θ, cache.p)
                end
                l
            end
            function (res, θ, σ, μ)
                FD.finite_difference_hessian!(h,
                    (x) -> lag(x, σ, μ),
                    θ,
                    updatecache(lag_hess_cache, θ))
                k = 1
                for i in 1:length(cache.u0), j in i:length(cache.u0)
                    res[k] = h[i, j]
                    k += 1
                end
            end
        end
    else
        lag_h = (res, θ, σ, μ) -> f.lag_h(res, θ, σ, μ, cache.p)
    end
    return OptimizationFunction{true}(f, adtype; grad = grad, hess = hess, hv = hv,
        cons = cons, cons_j = cons_j, cons_h = cons_h,
        cons_jac_colorvec = f.cons_jac_colorvec,
        hess_prototype = f.hess_prototype,
        cons_jac_prototype = f.cons_jac_prototype,
        cons_hess_prototype = f.cons_hess_prototype,
        lag_h, f.lag_hess_prototype)
end

end
