module OptimizationReverseDiffExt

import Optimization
import Optimization.SciMLBase: OptimizationFunction
import Optimization.ADTypes: AutoReverseDiff
# using SparseDiffTools, Symbolics
isdefined(Base, :get_extension) ? (using ReverseDiff, ReverseDiff.ForwardDiff) :
(using ..ReverseDiff, ..ReverseDiff.ForwardDiff)

function Optimization.instantiate_function(f, x, adtype::AutoReverseDiff,
    p = SciMLBase.NullParameters(),
    num_cons = 0)
    _f = (θ, args...) -> first(f.f(θ, p, args...))

    if f.grad === nothing
        cfg = ReverseDiff.GradientConfig(x)
        grad = (res, θ, args...) -> ReverseDiff.gradient!(res, x -> _f(x, args...), θ, cfg)
    else
        grad = (G, θ, args...) -> f.grad(G, θ, p, args...)
    end

    if f.hess === nothing
        hess = function (res, θ, args...)
            
            res .= SparseDiffTools.forwarddiff_color_jacobian(θ, colorvec = hess_colors, sparsity = hess_sparsity) do θ
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

    if f.cons === nothing
        cons = nothing
    else
        cons = (res, θ) -> f.cons(res, θ, p)
        cons_oop = (x) -> (_res = zeros(eltype(x), num_cons); cons(_res, x); _res)
    end

    if cons !== nothing && f.cons_j === nothing
        cjconfig = ReverseDiff.JacobianConfig(x)
        cons_j = function (J, θ)
            ReverseDiff.jacobian!(J, cons_oop, θ, cjconfig)
        end
    else
        cons_j = (J, θ) -> f.cons_j(J, θ, p)
    end

    if cons !== nothing && f.cons_h === nothing
        
        cons_h = function (res, θ)
            for i in 1:num_cons
                res[i] .= SparseDiffTools.forwarddiff_color_jacobian(θ, ) do θ
                    ReverseDiff.gradient(fncs[i], θ)
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
        hess_prototype = f.hess_prototype,
        cons_jac_prototype = f.cons_jac_prototype,
        cons_hess_prototype = f.cons_hess_prototype,
        lag_h, f.lag_hess_prototype)
end

# function Optimization.instantiate_function(f, cache::Optimization.ReInitCache,
#     adtype::AutoReverseDiff, num_cons = 0)
#     _f = (θ, args...) -> first(f.f(θ, cache.p, args...))

#     if f.grad === nothing
#         grad = (res, θ, args...) -> ReverseDiff.gradient!(res, x -> _f(x, args...), θ)
#     else
#         grad = (G, θ, args...) -> f.grad(G, θ, cache.p, args...)
#     end

#     if f.hess === nothing
#         hess_sparsity = Symbolics.hessian_sparsity(_f, cache.u0)
#         hess_colors = SparseDiffTools.matrix_colors(tril(hess_sparsity))
#         hess = function (res, θ, args...)
#             res .= SparseDiffTools.forwarddiff_color_jacobian(θ, colorvec = hess_colors, sparsity = hess_sparsity) do θ
#                 ReverseDiff.gradient(x -> _f(x, args...), θ)
#             end
#         end
#     else
#         hess = (H, θ, args...) -> f.hess(H, θ, cache.p, args...)
#     end

#     if f.hv === nothing
#         hv = function (H, θ, v, args...)
#             _θ = ForwardDiff.Dual.(θ, v)
#             res = similar(_θ)
#             grad(res, _θ, args...)
#             H .= getindex.(ForwardDiff.partials.(res), 1)
#         end
#     else
#         hv = f.hv
#     end

#     if f.cons === nothing
#         cons = nothing
#     else
#         cons = (res, θ) -> f.cons(res, θ, cache.p)
#         cons_oop = (x) -> (_res = zeros(eltype(x), num_cons); cons(_res, x); _res)
#     end

#     if cons !== nothing && f.cons_j === nothing
#         cjconfig = ReverseDiff.JacobianConfig(cache.u0)
#         cons_j = function (J, θ)
#             ReverseDiff.jacobian!(J, cons_oop, θ, cjconfig)
#         end
#     else
#         cons_j = (J, θ) -> f.cons_j(J, θ, cache.p)
#     end

#     if cons !== nothing && f.cons_h === nothing
#         fncs = [(x) -> cons_oop(x)[i] for i in 1:num_cons]
#         conshess_sparsity = Symbolics.hessian_sparsity.(fncs, Ref(cache.u0))
#         conshess_colors = SparseDiffTools.matrix_colors.(conshess_sparsity)
#         cons_h = function (res, θ)
#             for i in 1:num_cons
#                 res[i] .= SparseDiffTools.forwarddiff_color_jacobian(θ, colorvec = conshess_colors[i], sparsity = conshess_sparsity[i]) do θ
#                     ReverseDiff.gradient(fncs[i], θ)
#                 end
#             end
#         end
#     else
#         cons_h = (res, θ) -> f.cons_h(res, θ, cache.p)
#     end

#     if f.lag_h === nothing
#         lag_h = nothing # Consider implementing this
#     else
#         lag_h = (res, θ, σ, μ) -> f.lag_h(res, θ, σ, μ, cache.p)
#     end

#     return OptimizationFunction{true}(f.f, adtype; grad = grad, hess = hess, hv = hv,
#         cons = cons, cons_j = cons_j, cons_h = cons_h,
#         hess_prototype = f.hess_prototype,
#         cons_jac_prototype = f.cons_jac_prototype,
#         cons_hess_prototype = f.cons_hess_prototype,
#         lag_h, f.lag_hess_prototype)
# end

end
