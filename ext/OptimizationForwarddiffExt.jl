module OptimizationForwarddiffExt

import SciMLBase: OptimizationFunction, AbstractADType
import Optimization, ArrayInterface
import ADTypes: AutoForwardDiff
isdefined(Base, :get_extension) ? (using ForwardDiff) : (using ..ForwardDiff)

function default_chunk_size(len)
    if len < ForwardDiff.DEFAULT_CHUNK_THRESHOLD
        len
    else
        ForwardDiff.DEFAULT_CHUNK_THRESHOLD
    end
end

function Optimization.instantiate_function(f::OptimizationFunction{true}, x,
                                           adtype::AutoForwardDiff{_chunksize}, p,
                                           num_cons = 0) where {_chunksize}
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
        hesscfg = ForwardDiff.HessianConfig(_f, x, ForwardDiff.Chunk{chunksize}())
        hess = (res, θ, args...) -> ForwardDiff.hessian!(res, x -> _f(x, args...), θ,
                                                         hesscfg, Val{false}())
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
        cons = (res, θ) -> f.cons(res, θ, p)
        cons_oop = (x) -> (_res = zeros(eltype(x), num_cons); cons(_res, x); _res)
    end

    if cons !== nothing && f.cons_j === nothing
        cjconfig = ForwardDiff.JacobianConfig(cons_oop, x, ForwardDiff.Chunk{chunksize}())
        cons_j = function (J, θ)
            ForwardDiff.jacobian!(J, cons_oop, θ, cjconfig)
        end
    else
        cons_j = (J, θ) -> f.cons_j(J, θ, p)
    end

    if cons !== nothing && f.cons_h === nothing
        fncs = [(x) -> cons_oop(x)[i] for i in 1:num_cons]
        hess_config_cache = [ForwardDiff.HessianConfig(fncs[i], x,
                                                       ForwardDiff.Chunk{chunksize}())
                             for i in 1:num_cons]
        cons_h = function (res, θ)
            for i in 1:num_cons
                ForwardDiff.hessian!(res[i], fncs[i], θ, hess_config_cache[i], Val{true}())
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
                                           adtype::AutoForwardDiff{_chunksize},
                                           num_cons = 0) where {_chunksize}
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
        hesscfg = ForwardDiff.HessianConfig(_f, cache.u0, ForwardDiff.Chunk{chunksize}())
        hess = (res, θ, args...) -> ForwardDiff.hessian!(res, x -> _f(x, args...), θ,
                                                         hesscfg, Val{false}())
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
        cons = (res, θ) -> f.cons(res, θ, cache.p)
        cons_oop = (x) -> (_res = zeros(eltype(x), num_cons); cons(_res, x); _res)
    end

    if cons !== nothing && f.cons_j === nothing
        cjconfig = ForwardDiff.JacobianConfig(cons_oop, cache.u0,
                                              ForwardDiff.Chunk{chunksize}())
        cons_j = function (J, θ)
            ForwardDiff.jacobian!(J, cons_oop, θ, cjconfig)
        end
    else
        cons_j = (J, θ) -> f.cons_j(J, θ, cache.p)
    end

    if cons !== nothing && f.cons_h === nothing
        fncs = [(x) -> cons_oop(x)[i] for i in 1:num_cons]
        hess_config_cache = [ForwardDiff.HessianConfig(fncs[i], cache.u0,
                                                       ForwardDiff.Chunk{chunksize}())
                             for i in 1:num_cons]
        cons_h = function (res, θ)
            for i in 1:num_cons
                ForwardDiff.hessian!(res[i], fncs[i], θ, hess_config_cache[i], Val{true}())
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
