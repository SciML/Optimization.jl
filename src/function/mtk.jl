struct AutoModelingToolkit <: AbstractADType
    obj_sparse::Bool
    cons_sparse::Bool
end

AutoModelingToolkit() = AutoModelingToolkit(false, false)

function instantiate_function(f, x, adtype::AutoModelingToolkit, p, num_cons = 0)
    p = isnothing(p) ? SciMLBase.NullParameters() : p

    sys = ModelingToolkit.modelingtoolkitize(OptimizationProblem(f, x, p;
                                                                 lcons = fill(0.0,
                                                                              num_cons),
                                                                 ucons = fill(0.0,
                                                                              num_cons)))
    sys = ModelingToolkit.structural_simplify(sys)
    f = OptimizationProblem(sys, x, p, grad = true, hess = true,
                            sparse = adtype.obj_sparse, cons_j = true, cons_h = true,
                            cons_sparse = adtype.cons_sparse).f

    grad = (G, θ, args...) -> f.grad(G, θ, p, args...)

    hess = (H, θ, args...) -> f.hess(H, θ, p, args...)

    hv = function (H, θ, v, args...)
        res = adtype.obj_sparse ? hess_prototype : ArrayInterfaceCore.zeromatrix(θ)
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
                                      expr = symbolify(f.expr),
                                      cons_expr = symbolify.(f.cons_expr))
end
