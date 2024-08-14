module OptimizationNLPModels

using Reexport
@reexport using NLPModels, Optimization, ADTypes

function SciMLBase.OptimizationFunction(nlpmodel::AbstractNLPModel,
        adtype::ADTypes.AbstractADType = SciMLBase.NoAD(); kwargs...)
    f(x, p) = NLPModels.obj(nlpmodel, x)
    grad(G, u, p) = NLPModels.grad!(nlpmodel, u, G)
    hess(H, u, p) = (H .= NLPModels.hess(nlpmodel, u))
    hv(Hv, u, v, p) = NLPModels.hprod!(nlpmodel, u, v, Hv)

    if !unconstrained(nlpmodel)
        cons(res, x, p) = NLPModels.cons!(nlpmodel, x, res)
        cons_j(J, x, p) = (J .= NLPModels.jac(nlpmodel, x))
        cons_jvp(Jv, v, x, p) = NLPModels.jprod!(nlpmodel, x, v, Jv)
        lag_h(x, sigma, mu, p) = NLPModels.hess(nlpmodel, x, mu; obj_weight = sigma)

        return OptimizationFunction(
            f, adtype; grad, hess, hv, cons, cons_j, cons_jvp, lag_h, kwargs...)
    end

    return OptimizationFunction(f, adtype; grad, hess, hv, kwargs...)
end

function OptimizationProblem(nlpmodel::AbstractNLPModel,
        adtype::ADTypes.AbstractADType = SciMLBase.NoAD(); kwargs...)
    f = OptimizationFunction(nlpmodel, kwargs...)
    u0 = nlpmodel.meta.x0
    lb = nlpmodel.meta.lvar
    ub = nlpmodel.meta.uvar
    lcons = nlpmodel.meta.lcon
    ucons = nlpmodel.meta.ucon
    sense = nlpmodel.meta.minimize ? Optimization.MinSense : Optimization.MaxSense

    # The number of variables, geometry of u0, etc.. are valid and were checked when the
    # nlpmodel was created.

    return Optimization.OptimizationProblem(
        f, u0; lb = lb, ub = ub, lcons = lcons, ucons = ucons, sense = sense, kwargs...)
end

end
