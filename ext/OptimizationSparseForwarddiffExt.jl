module OptimizationSparseForwarddiffExt

import SciMLBase: OptimizationFunction
import Optimization, ArrayInterface
import ADTypes: AutoSparseForwardDiff
import Symbolics
using LinearAlgebra
isdefined(Base, :get_extension) ? (using ForwardDiff, SparseDiffTools) :
(using ..ForwardDiff, ..SparseDiffTools)

function default_chunk_size(len)
    if len < ForwardDiff.DEFAULT_CHUNK_THRESHOLD
        len
    else
        ForwardDiff.DEFAULT_CHUNK_THRESHOLD
    end
end

function Optimization.instantiate_function(f::OptimizationFunction{true}, x,
    adtype::AutoSparseForwardDiff{_chunksize}, p,
    num_cons = 0) where {_chunksize}
    if maximum(getfield.(methods(f.f), :nargs)) > 2
        error("$(string(adtype)) with SparseDiffTools does not support functions with more than 2 arguments")
    end
    chunksize = _chunksize === nothing ? default_chunk_size(length(x)) : _chunksize

    _f = (θ, args...) -> first(f.f(θ, p, args...))

    if f.grad === nothing
        gradcfg = ForwardDiff.GradientConfig(_f, x, ForwardDiff.Chunk{chunksize}())
        grad = (res, θ, args...) -> ForwardDiff.gradient!(res, x -> _f(x, args...), θ,
            gradcfg, Val{false}())
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
            num_hesvecgrad!(H, (res, x) -> grad(res, x, args...), θ, v)
        end
    else
        hv = f.hv
    end

    if f.cons === nothing
        cons = nothing
    else
        cons = (res, θ) -> f.cons(res, θ, p)
        cons_oop = (x) -> (_res = zeros(eltype(x), num_cons); cons(_res, x); _res)
    end

    if cons !== nothing && f.cons_j === nothing
        cons_jac_prototype = Symbolics.jacobian_sparsity(cons, zeros(eltype(x), num_cons),
            x)
        cons_jac_colorvec = matrix_colors(tril(cons_jac_prototype))
        jaccache = ForwardColorJacCache(cons,
            x,
            chunksize;
            colorvec = cons_jac_colorvec,
            sparsity = cons_jac_prototype)
        cons_j = function (J, θ)
            forwarddiff_color_jacobian!(J, cons, θ, jaccache)
        end
    else
        cons_j = (J, θ) -> f.cons_j(J, θ, p)
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

function Optimization.instantiate_function(f::OptimizationFunction{true},
    cache::Optimization.ReInitCache,
    adtype::AutoSparseForwardDiff{_chunksize},
    num_cons = 0) where {_chunksize}
    if maximum(getfield.(methods(f.f), :nargs)) > 2
        error("$(string(adtype)) with SparseDiffTools does not support functions with more than 2 arguments")
    end
    chunksize = _chunksize === nothing ? default_chunk_size(length(cache.u0)) : _chunksize

    _f = (θ, args...) -> first(f.f(θ, cache.p, args...))

    if f.grad === nothing
        gradcfg = ForwardDiff.GradientConfig(_f, cache.u0, ForwardDiff.Chunk{chunksize}())
        grad = (res, θ, args...) -> ForwardDiff.gradient!(res, x -> _f(x, args...), θ,
            gradcfg, Val{false}())
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
            num_hesvecgrad!(H, (res, x) -> grad(res, x, args...), θ, v)
        end
    else
        hv = f.hv
    end

    if f.cons === nothing
        cons = nothing
    else
        cons = (res, θ) -> f.cons(res, θ, cache.p)
        cons_oop = (x) -> (_res = zeros(eltype(x), num_cons); cons(_res, x); _res)
    end

    if cons !== nothing && f.cons_j === nothing
        cons_jac_prototype = Symbolics.jacobian_sparsity(cons,
            zeros(eltype(cache.u0), num_cons),
            cache.u0)
        cons_jac_colorvec = matrix_colors(tril(cons_jac_prototype))
        jaccache = ForwardColorJacCache(cons, cache.u0, chunksize;
            colorvec = cons_jac_colorvec,
            sparsity = cons_jac_prototype)
        cons_j = function (J, θ)
            forwarddiff_color_jacobian!(J, cons, θ, jaccache)
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
