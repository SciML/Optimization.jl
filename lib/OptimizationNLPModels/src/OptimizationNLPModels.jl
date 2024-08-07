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

function OptimizationProblem(nlpmodel::NLPModel{T, S})
end

end
