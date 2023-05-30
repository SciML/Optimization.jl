module OptimizationMTKExt

import SciMLBase
import SciMLBase: OptimizationFunction, AbstractADType
import Optimization, ArrayInterface
import ADTypes: AutoModelingToolkit
isdefined(Base, :get_extension) ? (using ModelingToolkit) : (using ..ModelingToolkit)

function Optimization.instantiate_function(f, x, adtype::AutoModelingToolkit, p,
                                           num_cons = 0)
    p = isnothing(p) ? SciMLBase.NullParameters() : p

    sys = ModelingToolkit.modelingtoolkitize(OptimizationProblem(f, x, p;
                                                                 lcons = fill(0.0,
                                                                              num_cons),
                                                                 ucons = fill(0.0,
                                                                              num_cons)))
    #sys = ModelingToolkit.structural_simplify(sys)
    f = OptimizationProblem(sys, x, p, grad = true, hess = true,
                            sparse = adtype.obj_sparse, cons_j = true, cons_h = true,
                            cons_sparse = adtype.cons_sparse).f

    grad = (G, θ, args...) -> f.grad(G, θ, p, args...)

    hess = (H, θ, args...) -> f.hess(H, θ, p, args...)

    hv = function (H, θ, v, args...)
        res = adtype.obj_sparse ? (eltype(θ)).(f.hess_prototype) :
              ArrayInterface.zeromatrix(θ)
        hess(res, θ, args...)
        H .= res * v
    end

    cons = (res, θ) -> f.cons(res, θ, p)

    cons_j = (J, θ) -> f.cons_j(J, θ, p)

    cons_h = (res, θ) -> f.cons_h(res, θ, p)
    return OptimizationFunction{true}(f.f, adtype; grad = grad, hess = hess, hv = hv,
                                      cons = cons, cons_j = cons_j, cons_h = cons_h,
                                      hess_prototype = f.hess_prototype,
                                      cons_jac_prototype = f.cons_jac_prototype,
                                      cons_hess_prototype = f.cons_hess_prototype,
                                      expr = Optimization.symbolify(f.expr),
                                      cons_expr = Optimization.symbolify.(f.cons_expr),
                                      observed = f.observed)
end

function Optimization.instantiate_function(f, cache::Optimization.ReInitCache,
                                           adtype::AutoModelingToolkit, num_cons = 0)
    p = isnothing(cache.p) ? SciMLBase.NullParameters() : cache.p

    sys = ModelingToolkit.modelingtoolkitize(OptimizationProblem(f, cache.u0, cache.p;
                                                                 lcons = fill(0.0,
                                                                              num_cons),
                                                                 ucons = fill(0.0,
                                                                              num_cons)))
    #sys = ModelingToolkit.structural_simplify(sys)
    f = OptimizationProblem(sys, cache.u0, cache.p, grad = true, hess = true,
                            sparse = adtype.obj_sparse, cons_j = true, cons_h = true,
                            cons_sparse = adtype.cons_sparse).f

    grad = (G, θ, args...) -> f.grad(G, θ, cache.p, args...)

    hess = (H, θ, args...) -> f.hess(H, θ, cache.p, args...)

    hv = function (H, θ, v, args...)
        res = adtype.obj_sparse ? (eltype(θ)).(f.hess_prototype) :
              ArrayInterface.zeromatrix(θ)
        hess(res, θ, args...)
        H .= res * v
    end

    cons = (res, θ) -> f.cons(res, θ, cache.p)

    cons_j = (J, θ) -> f.cons_j(J, θ, cache.p)

    cons_h = (res, θ) -> f.cons_h(res, θ, cache.p)
    return OptimizationFunction{true}(f.f, adtype; grad = grad, hess = hess, hv = hv,
                                      cons = cons, cons_j = cons_j, cons_h = cons_h,
                                      hess_prototype = f.hess_prototype,
                                      cons_jac_prototype = f.cons_jac_prototype,
                                      cons_hess_prototype = f.cons_hess_prototype,
                                      expr = Optimization.symbolify(f.expr),
                                      cons_expr = Optimization.symbolify.(f.cons_expr),
                                      observed = f.observed)
end

end
