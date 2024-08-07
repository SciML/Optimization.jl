module OptimizationNLPModels

using Reexport
@reexport using NLPModels, Optimization
using Optimization.SciMLBase

function OptimizationFunction(
        nlpmodel::NLPModel, adtype::AbstractADType = NoAD(); kwargs...)
    f(x, p) = NLPModels.obj(nlpmodel, x)
    grad(G, u, p) = NLPModels.grad!(nlpmodel, u, G)
    hess(u, p) = NLPModels.hess(nlpmodel, u)
    hv(Hv, u, v, p) = NLPModels.hprod!(nlpmodel, u, v, Hv)
    cons(res, x, p) = NLPModels.cons!(nlpmodel, x, res)
    cons_j(x, p) = NLPModels.jac(nlpmodel, x)
    cons_jvp(Jv, v, x, p) = NLPModels.jprod!(nlpmodel, x, v, Jv)
    hess_prototype = SparseMatrixCSC
    cons_jac_prototype = SparseMatrixCSC
    lag_h(x, sigma, mu, p) = NLPModels.hess(nlpmodel, x, mu; obj_weight = sigma)

    return OptimizationFunction(f, adtype; grad, hess, hv, cons, cons_j, cons_jvp,
        hess_prototype, cons_jac_prototype, lag_h, kwargs...)
end

function OptimizationProblem(nlpmodel::NLPModel, adtype::AbstractADType = NoAD(); kwargs...)
    f = OptimizationFunction(nlpmodel, adtype, kwargs...)
    # FIXME: Check lengths
    u0 = nlp.meta.x0
    lb = nlp.meta.lvar
    ub = nlp.meta.uvar
    lcons = nlp.meta.lcon
    ucons = nlp.meta.ucon
    sense = nlp.meta.minimize ? Optimization.MinSense() : Optimization.MaxSense()

    n = length(u0)
    @assert n == nlp.meta.nvar
    err_intro = "Error converting `NLPModel: $(nlpmodel) "

    length(lb) != n && error(err_intro * "More lower bounds than variables were given")
    length(ub) != n && error(err_intro * "More upper bounds than variables were given")
    length(lcons) != n &&
        error(err_intro * "More inequality lower bounds than variables were given")
    length(ucons) != n &&
        error(err_intro * "More inequality upper bounds than variables were given")

    return OptimizationProblem(
        f, u0, lb, ub, lcons, ucons, sense; prob.kwargs..., kwargs...)
end

end
