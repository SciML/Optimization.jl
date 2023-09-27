module OptimizationSparseDiffExt

import Optimization, Optimization.ArrayInterface
import Optimization.SciMLBase: OptimizationFunction
import Optimization.ADTypes: AutoSparseForwardDiff, AutoSparseFiniteDiff, AutoSparseReverseDiff
using Optimization.LinearAlgebra, ReverseDiff
isdefined(Base, :get_extension) ? (using SparseDiffTools, SparseDiffTools.ForwardDiff, SparseDiffTools.FiniteDiff, Symbolics) :
(using ..SparseDiffTools, ..SparseDiffTools.ForwardDiff, ..SparseDiffTools.FiniteDiff, ..Symbolics)

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
    if maximum(getfield.(methods(f.f), :nargs)) > 3
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

    hess_sparsity = f.hess_prototype
    hess_colors = f.hess_colorvec
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

    cons_jac_prototype = f.cons_jac_prototype
    cons_jac_colorvec = f.cons_jac_colorvec
    if cons !== nothing && f.cons_j === nothing
        cons_jac_prototype = Symbolics.jacobian_sparsity(cons, zeros(eltype(x), num_cons),
            x)
        cons_jac_colorvec = matrix_colors(cons_jac_prototype)
        jaccache = ForwardColorJacCache(cons,
            x,
            chunksize;
            colorvec = cons_jac_colorvec,
            sparsity = cons_jac_prototype,
            dx = zeros(eltype(x), num_cons))
        cons_j = function (J, θ)
            forwarddiff_color_jacobian!(J, cons, θ, jaccache)
        end
    else
        cons_j = (J, θ) -> f.cons_j(J, θ, p)
    end

    cons_hess_caches = [(; sparsity = f.cons_hess_prototype, colors = f.cons_hess_colorvec)]
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
        cons_hess_caches = gen_conshess_cache.(fcons, Ref(x))
        cons_h = function (res, θ)
            for i in 1:num_cons
                numauto_color_hessian!(res[i], fcons[i], θ, cons_hess_caches[i])
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
        hess_prototype = hess_sparsity,
        hess_colorvec = hess_colors,
        cons_jac_colorvec = cons_jac_colorvec,
        cons_jac_prototype = cons_jac_prototype,
        cons_hess_prototype = getfield.(cons_hess_caches, :sparsity),
        cons_hess_colorvec = getfield.(cons_hess_caches, :colors),
        lag_h, f.lag_hess_prototype)
end

function Optimization.instantiate_function(f::OptimizationFunction{true},
    cache::Optimization.ReInitCache,
    adtype::AutoSparseForwardDiff{_chunksize},
    num_cons = 0) where {_chunksize}
    if maximum(getfield.(methods(f.f), :nargs)) > 3
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

    hess_sparsity = f.hess_prototype
    hess_colors = f.hess_colorvec
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

    cons_jac_prototype = f.cons_jac_prototype
    cons_jac_colorvec = f.cons_jac_colorvec
    if cons !== nothing && f.cons_j === nothing
        cons_jac_prototype = Symbolics.jacobian_sparsity(cons,
            zeros(eltype(cache.u0), num_cons),
            cache.u0)
        cons_jac_colorvec = matrix_colors(cons_jac_prototype)
        jaccache = ForwardColorJacCache(cons, cache.u0, chunksize;
            colorvec = cons_jac_colorvec,
            sparsity = cons_jac_prototype,
            dx = zeros(eltype(cache.u0), num_cons))
        cons_j = function (J, θ)
            forwarddiff_color_jacobian!(J, cons, θ, jaccache)
        end
    else
        cons_j = (J, θ) -> f.cons_j(J, θ, cache.p)
    end

    cons_hess_caches = [(; sparsity = f.cons_hess_prototype, colors = f.cons_hess_colorvec)]
    if cons !== nothing && f.cons_h === nothing
        function gen_conshess_cache(_f, x)
            conshess_sparsity = copy(Symbolics.hessian_sparsity(_f, x))
            conshess_colors = matrix_colors(tril(conshess_sparsity))
            hesscache = ForwardColorHesCache(_f, x, conshess_colors,
                conshess_sparsity)
            return hesscache
        end

        fcons = [(x) -> (_res = zeros(eltype(x), num_cons);
        cons(_res, x);
        _res[i]) for i in 1:num_cons]
        cons_hess_caches = gen_conshess_cache.(fcons, Ref(cache.u0))
        cons_h = function (res, θ)
            for i in 1:num_cons
                numauto_color_hessian!(res[i], fcons[i], θ, cons_hess_caches[i])
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
        hess_prototype = hess_sparsity,
        hess_colorvec = hess_colors,
        cons_jac_prototype = cons_jac_prototype,
        cons_jac_colorvec = cons_jac_colorvec,
        cons_hess_prototype = getfield.(cons_hess_caches, :sparsity),
        cons_hess_colorvec = getfield.(cons_hess_caches, :colors),
        lag_h, f.lag_hess_prototype)
end

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
    
    hess_sparsity = f.hess_prototype
    hess_colors = f.hess_colorvec
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

    cons_jac_prototype = f.cons_jac_prototype
    cons_jac_colorvec = f.cons_jac_colorvec
    if cons !== nothing && f.cons_j === nothing
        cons_jac_prototype = f.cons_jac_prototype === nothing ?
                             Symbolics.jacobian_sparsity(cons,
            zeros(eltype(x), num_cons),
            x) :
                             f.cons_jac_prototype
        cons_jac_colorvec = f.cons_jac_colorvec === nothing ?
                            matrix_colors(cons_jac_prototype) :
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

    conshess_caches = [(; sparsity = f.cons_hess_prototype, colors = f.cons_hess_colorvec)]
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
        conshess_caches = gen_conshess_cache.(fcons, Ref(x))
        cons_h = function (res, θ)
            for i in 1:num_cons
                numauto_color_hessian!(res[i], fcons[i], θ, conshess_caches[i])
            end
        end
    else
        cons_h = (res, θ) -> f.cons_h(res, θ, p)
    end

    if f.lag_h === nothing
        # lag_hess_cache = FD.HessianCache(copy(x))
        # c = zeros(num_cons)
        # h = zeros(length(x), length(x))
        # lag_h = let c = c, h = h
        #     lag = function (θ, σ, μ)
        #         f.cons(c, θ, p)
        #         l = μ'c
        #         if !iszero(σ)
        #             l += σ * f.f(θ, p)
        #         end
        #         l
        #     end
        #     function (res, θ, σ, μ)
        #         FD.finite_difference_hessian!(res,
        #             (x) -> lag(x, σ, μ),
        #             θ,
        #             updatecache(lag_hess_cache, θ))
        #     end
        # end
        lag_h = nothing
    else
        lag_h = (res, θ, σ, μ) -> f.lag_h(res, θ, σ, μ, p)
    end
    return OptimizationFunction{true}(f, adtype; grad = grad, hess = hess, hv = hv,
        cons = cons, cons_j = cons_j, cons_h = cons_h,
        hess_prototype = hess_sparsity,
        hess_colorvec = hess_colors,
        cons_jac_prototype = cons_jac_prototype,
        cons_jac_colorvec = cons_jac_colorvec,
        cons_hess_prototype = getfield.(conshess_caches, :sparsity),
        cons_hess_colorvec = getfield.(conshess_caches, :colors),
        lag_h, f.lag_hess_prototype)
end

function Optimization.instantiate_function(f, cache::Optimization.ReInitCache,
    adtype::AutoSparseFiniteDiff, num_cons = 0)
    if maximum(getfield.(methods(f.f), :nargs)) > 3
        error("$(string(adtype)) with SparseDiffTools does not support functions with more than 2 arguments")
    end
    _f = (θ, args...) -> first(f.f(θ, cache.p, args...))

    if f.grad === nothing
        gradcache = FD.GradientCache(cache.u0, cache.u0)
        grad = (res, θ, args...) -> FD.finite_difference_gradient!(res, x -> _f(x, args...),
            θ, gradcache)
    else
        grad = (G, θ, args...) -> f.grad(G, θ, cache.p, args...)
    end
    
    hess_sparsity = f.hess_prototype
    hess_colors = f.hess_colorvec
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

    cons_jac_prototype = f.cons_jac_prototype
    cons_jac_colorvec = f.cons_jac_colorvec
    if cons !== nothing && f.cons_j === nothing
        cons_jac_prototype = f.cons_jac_prototype === nothing ?
                             Symbolics.jacobian_sparsity(cons, zeros(eltype(cache.u0), num_cons),
            cache.u0) :
                             f.cons_jac_prototype
        cons_jac_colorvec = f.cons_jac_colorvec === nothing ?
                            matrix_colors(cons_jac_prototype) :
                            f.cons_jac_colorvec
        cons_j = function (J, θ)
            y0 = zeros(num_cons)
            jaccache = FD.JacobianCache(copy(θ), copy(y0), copy(y0);
                colorvec = cons_jac_colorvec,
                sparsity = cons_jac_prototype)
            FD.finite_difference_jacobian!(J, cons, θ, jaccache)
        end
    else
        cons_j = (J, θ) -> f.cons_j(J, θ, cache.p)
    end

    conshess_caches = [(; sparsity = f.cons_hess_prototype, colors = f.cons_hess_colorvec)]
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
        conshess_caches = [gen_conshess_cache(fcons[i], cache.u0) for i in 1:num_cons]
        cons_h = function (res, θ)
            for i in 1:num_cons
                numauto_color_hessian!(res[i], fcons[i], θ, conshess_caches[i])
            end
        end
    else
        cons_h = (res, θ) -> f.cons_h(res, θ, cache.p)
    end
    if f.lag_h === nothing
        # lag_hess_cache = FD.HessianCache(copy(cache.u0))
        # c = zeros(num_cons)
        # h = zeros(length(cache.u0), length(cache.u0))
        # lag_h = let c = c, h = h
        #     lag = function (θ, σ, μ)
        #         f.cons(c, θ, cache.p)
        #         l = μ'c
        #         if !iszero(σ)
        #             l += σ * f.f(θ, cache.p)
        #         end
        #         l
        #     end
        #     function (res, θ, σ, μ)
        #         FD.finite_difference_hessian!(h,
        #             (x) -> lag(x, σ, μ),
        #             θ,
        #             updatecache(lag_hess_cache, θ))
        #         k = 1
        #         for i in 1:length(cache.u0), j in i:length(cache.u0)
        #             res[k] = h[i, j]
        #             k += 1
        #         end
        #     end
        # end
        lag_h = nothing
    else
        lag_h = (res, θ, σ, μ) -> f.lag_h(res, θ, σ, μ, cache.p)
    end
    return OptimizationFunction{true}(f, adtype; grad = grad, hess = hess, hv = hv,
        cons = cons, cons_j = cons_j, cons_h = cons_h,
        hess_prototype = hess_sparsity,
        hess_colorvec = hess_colors,
        cons_jac_prototype = cons_jac_prototype,
        cons_jac_colorvec = cons_jac_colorvec,
        cons_hess_prototype = getfield.(conshess_caches, :sparsity),
        cons_hess_colorvec = getfield.(conshess_caches, :colors),
        lag_h, f.lag_hess_prototype)
end

struct OptimizationSparseReverseTag end

function Optimization.instantiate_function(f, x, adtype::AutoSparseReverseDiff,
    p = SciMLBase.NullParameters(),
    num_cons = 0)
    _f = (θ, args...) -> first(f.f(θ, p, args...))

    chunksize = default_chunk_size(length(x))

    if f.grad === nothing
        if adtype.compile
            _tape = ReverseDiff.GradientTape(_f, x)
            tape = ReverseDiff.compile(_tape)
            grad = function (res, θ, args...)
                ReverseDiff.gradient!(res, tape, θ)
            end
        else
            cfg = ReverseDiff.GradientConfig(x)
            grad = (res, θ, args...) -> ReverseDiff.gradient!(res, x -> _f(x, args...), θ, cfg)
        end
    else
        grad = (G, θ, args...) -> f.grad(G, θ, p, args...)
    end

    hess_sparsity = f.hess_prototype
    hess_colors = f.hess_colorvec
    if f.hess === nothing
        hess_sparsity = Symbolics.hessian_sparsity(_f, x)
        hess_colors = SparseDiffTools.matrix_colors(tril(hess_sparsity))
        if adtype.compile
            T = ForwardDiff.Tag(OptimizationSparseReverseTag(),eltype(x))
            xdual = ForwardDiff.Dual{typeof(T),eltype(x),min(chunksize, maximum(hess_colors))}.(x, Ref(ForwardDiff.Partials((ones(eltype(x), min(chunksize, maximum(hess_colors)))...,))))
            h_tape = ReverseDiff.GradientTape(_f, xdual)
            htape = ReverseDiff.compile(h_tape)
            function g(res1, θ)
                ReverseDiff.gradient!(res1, htape, θ)
            end
            jaccfg = ForwardColorJacCache(g, x; tag = typeof(T), colorvec = hess_colors, sparsity = hess_sparsity)
            hess = function (res, θ, args...)
                SparseDiffTools.forwarddiff_color_jacobian!(res, g, θ, jaccfg)
            end
        else
            hess = function (res, θ, args...)
                res .= SparseDiffTools.forwarddiff_color_jacobian(θ, colorvec = hess_colors, sparsity = hess_sparsity) do θ
                    ReverseDiff.gradient(x -> _f(x, args...), θ)
                end
            end
        end
    else
        hess = (H, θ, args...) -> f.hess(H, θ, p, args...)
    end

    if f.hv === nothing
        hv = function (H, θ, v, args...)
            # _θ = ForwardDiff.Dual.(θ, v)
            # res = similar(_θ)
            # grad(res, _θ, args...)
            # H .= getindex.(ForwardDiff.partials.(res), 1)
            res = zeros(length(θ), length(θ))
            hess(res, θ, args...)
            H .= res * v
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

    cons_jac_prototype = f.cons_jac_prototype
    cons_jac_colorvec = f.cons_jac_colorvec
    if cons !== nothing && f.cons_j === nothing
        cons_jac_prototype = Symbolics.jacobian_sparsity(cons,
            zeros(eltype(x), num_cons),
            x)
        cons_jac_colorvec = matrix_colors(cons_jac_prototype)
        jaccache = ForwardColorJacCache(cons, x;
            colorvec = cons_jac_colorvec,
            sparsity = cons_jac_prototype,
            dx = zeros(eltype(x), num_cons))
        cons_j = function (J, θ)
            forwarddiff_color_jacobian!(J, cons, θ, jaccache)
        end
    else
        cons_j = (J, θ) -> f.cons_j(J, θ, p)
    end
    
    conshess_sparsity = f.cons_hess_prototype
    conshess_colors = f.cons_hess_colorvec
    if cons !== nothing && f.cons_h === nothing
        fncs = [(x) -> cons_oop(x)[i] for i in 1:num_cons]
        conshess_sparsity = Symbolics.hessian_sparsity.(fncs, Ref(x))
        conshess_colors = SparseDiffTools.matrix_colors.(conshess_sparsity)
        if adtype.compile
            T = ForwardDiff.Tag(OptimizationSparseReverseTag(),eltype(x))
            xduals = [ForwardDiff.Dual{typeof(T),eltype(x),min(chunksize, maximum(conshess_colors[i]))}.(x, Ref(ForwardDiff.Partials((ones(eltype(x), min(chunksize, maximum(conshess_colors[i])))...,)))) for i in 1:num_cons]
            consh_tapes = [ReverseDiff.GradientTape(fncs[i], xduals[i]) for i in 1:num_cons] 
            conshtapes = ReverseDiff.compile.(consh_tapes)
            function grad_cons(res1, θ, htape)
                ReverseDiff.gradient!(res1, htape, θ)
            end
            gs = [(res1, x) -> grad_cons(res1, x, conshtapes[i]) for i in 1:num_cons]
            jaccfgs = [ForwardColorJacCache(gs[i], x; tag = typeof(T), colorvec = conshess_colors[i], sparsity = conshess_sparsity[i]) for i in 1:num_cons]
            cons_h = function (res, θ)
                for i in 1:num_cons
                    SparseDiffTools.forwarddiff_color_jacobian!(res[i], gs[i], θ, jaccfgs[i])
                end
            end
        else
            cons_h = function (res, θ)
                for i in 1:num_cons
                    res[i] .= SparseDiffTools.forwarddiff_color_jacobian(θ, colorvec = conshess_colors[i], sparsity = conshess_sparsity[i]) do θ
                        ReverseDiff.gradient(fncs[i], θ)
                    end
                end
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
        hess_prototype = hess_sparsity,
        hess_colorvec = hess_colors,
        cons_jac_prototype = cons_jac_prototype,
        cons_jac_colorvec = cons_jac_colorvec,
        cons_hess_prototype = conshess_sparsity,
        cons_hess_colorvec = conshess_colors,
        lag_h, f.lag_hess_prototype)
end

function Optimization.instantiate_function(f, cache::Optimization.ReInitCache,
    adtype::AutoSparseReverseDiff, num_cons = 0)
    _f = (θ, args...) -> first(f.f(θ, cache.p, args...))

    chunksize = default_chunk_size(length(cache.u0))

    if f.grad === nothing
        if adtype.compile
            _tape = ReverseDiff.GradientTape(_f, cache.u0)
            tape = ReverseDiff.compile(_tape)
            grad = function (res, θ, args...)
                ReverseDiff.gradient!(res, tape, θ)
            end
        else
            cfg = ReverseDiff.GradientConfig(cache.u0)
            grad = (res, θ, args...) -> ReverseDiff.gradient!(res, x -> _f(x, args...), θ, cfg)
        end
    else
        grad = (G, θ, args...) -> f.grad(G, θ, cache.p, args...)
    end

    hess_sparsity = f.hess_prototype
    hess_colors = f.hess_colorvec
    if f.hess === nothing
        hess_sparsity = Symbolics.hessian_sparsity(_f, cache.u0)
        hess_colors = SparseDiffTools.matrix_colors(tril(hess_sparsity))
        if adtype.compile
            T = ForwardDiff.Tag(OptimizationSparseReverseTag(),eltype(cache.u0))
            xdual = ForwardDiff.Dual{typeof(T),eltype(cache.u0),min(chunksize, maximum(hess_colors))}.(cache.u0, Ref(ForwardDiff.Partials((ones(eltype(cache.u0), min(chunksize, maximum(hess_colors)))...,))))
            h_tape = ReverseDiff.GradientTape(_f, xdual)
            htape = ReverseDiff.compile(h_tape)
            function g(res1, θ)
                ReverseDiff.gradient!(res1, htape, θ)
            end
            jaccfg = ForwardColorJacCache(g, cache.u0; tag = typeof(T), colorvec = hess_colors, sparsity = hess_sparsity)
            hess = function (res, θ, args...)
                SparseDiffTools.forwarddiff_color_jacobian!(res, g, θ, jaccfg)
            end
        else
            hess = function (res, θ, args...)
                res .= SparseDiffTools.forwarddiff_color_jacobian(θ, colorvec = hess_colors, sparsity = hess_sparsity) do θ
                    ReverseDiff.gradient(x -> _f(x, args...), θ)
                end
            end
        end
    else
        hess = (H, θ, args...) -> f.hess(H, θ, cache.p, args...)
    end

    if f.hv === nothing
        hv = function (H, θ, v, args...)
            # _θ = ForwardDiff.Dual.(θ, v)
            # res = similar(_θ)
            # grad(res, _θ, args...)
            # H .= getindex.(ForwardDiff.partials.(res), 1)
            res = zeros(length(θ), length(θ))
            hess(res, θ, args...)
            H .= res * v
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

    cons_jac_prototype = f.cons_jac_prototype
    cons_jac_colorvec = f.cons_jac_colorvec
    if cons !== nothing && f.cons_j === nothing
        cons_jac_prototype = Symbolics.jacobian_sparsity(cons,
            zeros(eltype(cache.u0), num_cons),
            cache.u0)
        cons_jac_colorvec = matrix_colors(cons_jac_prototype)
        jaccache = ForwardColorJacCache(cons, cache.u0;
            colorvec = cons_jac_colorvec,
            sparsity = cons_jac_prototype,
            dx = zeros(eltype(cache.u0), num_cons))
        cons_j = function (J, θ)
            forwarddiff_color_jacobian!(J, cons, θ, jaccache)
        end
    else
        cons_j = (J, θ) -> f.cons_j(J, θ, cache.p)
    end
    
    conshess_sparsity = f.cons_hess_prototype
    conshess_colors = f.cons_hess_colorvec
    if cons !== nothing && f.cons_h === nothing
        fncs = [(x) -> cons_oop(x)[i] for i in 1:num_cons]
        conshess_sparsity = Symbolics.hessian_sparsity.(fncs, Ref(cache.u0))
        conshess_colors = SparseDiffTools.matrix_colors.(conshess_sparsity)
        if adtype.compile
            T = ForwardDiff.Tag(OptimizationSparseReverseTag(),eltype(cache.u0))
            xduals = [ForwardDiff.Dual{typeof(T),eltype(cache.u0),min(chunksize, maximum(conshess_colors[i]))}.(cache.u0, Ref(ForwardDiff.Partials((ones(eltype(cache.u0), min(chunksize, maximum(conshess_colors[i])))...,)))) for i in 1:num_cons]
            consh_tapes = [ReverseDiff.GradientTape(fncs[i], xduals[i]) for i in 1:num_cons] 
            conshtapes = ReverseDiff.compile.(consh_tapes)
            function grad_cons(res1, θ, htape)
                ReverseDiff.gradient!(res1, htape, θ)
            end
            gs = [(res1, x) -> grad_cons(res1, x, conshtapes[i]) for i in 1:num_cons]
            jaccfgs = [ForwardColorJacCache(gs[i], cache.u0; tag = typeof(T), colorvec = conshess_colors[i], sparsity = conshess_sparsity[i]) for i in 1:num_cons]
            cons_h = function (res, θ)
                for i in 1:num_cons
                    SparseDiffTools.forwarddiff_color_jacobian!(res[i], gs[i], θ, jaccfgs[i])
                end
            end
        else
            cons_h = function (res, θ)
                for i in 1:num_cons
                    res[i] .= SparseDiffTools.forwarddiff_color_jacobian(θ, colorvec = conshess_colors[i], sparsity = conshess_sparsity[i]) do θ
                        ReverseDiff.gradient(fncs[i], θ)
                    end
                end
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
        hess_prototype = hess_sparsity,
        hess_colorvec = hess_colors,
        cons_jac_prototype = cons_jac_prototype,
        cons_jac_colorvec = cons_jac_colorvec,
        cons_hess_prototype = conshess_sparsity,
        cons_hess_colorvec = conshess_colors,
        lag_h, f.lag_hess_prototype)
end

end
