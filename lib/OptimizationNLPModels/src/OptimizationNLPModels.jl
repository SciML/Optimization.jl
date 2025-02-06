module OptimizationNLPModels

using Reexport, SparseArrays
@reexport using NLPModels, Optimization, ADTypes

"""
    OptimizationFunction(nlpmodel::AbstractNLPModel, adtype::AbstractADType = NoAD())

Returns an `OptimizationFunction` from the `NLPModel` defined in `nlpmodel` where the
available derivates are re-used from the model, and the rest are populated with the
Automatic Differentiation backend specified by `adtype`.
"""
function SciMLBase.OptimizationFunction(nlpmodel::AbstractNLPModel,
        adtype::ADTypes.AbstractADType = SciMLBase.NoAD(); kwargs...)
    f(x, p) = NLPModels.obj(nlpmodel, x)
    grad(G, u, p) = NLPModels.grad!(nlpmodel, u, G)
    hess(H, u, p) = (H .= NLPModels.hess(nlpmodel, u))
    hv(Hv, u, v, p) = NLPModels.hprod!(nlpmodel, u, v, Hv)

    if !unconstrained(nlpmodel) && !bound_constrained(nlpmodel)
        cons(res, x, p) = NLPModels.cons!(nlpmodel, x, res)
        cons_j(J, x, p) = (J .= NLPModels.jac(nlpmodel, x))
        cons_jvp(Jv, v, x, p) = NLPModels.jprod!(nlpmodel, x, v, Jv)
        function lag_h(h, θ, σ, λ)
            H = NLPModels.hess(nlpmodel, θ, λ; obj_weight = σ)
            k = 0
            rows, cols, _ = findnz(H)
            for (i, j) in zip(rows, cols)
                if i <= j
                    k += 1
                    h[k] = H[i, j]
                end
            end
        end

        return OptimizationFunction(
            f, adtype; grad, hess, hv, cons, cons_j, cons_jvp, lag_h, kwargs...)
    end

    return OptimizationFunction(f, adtype; grad, hess, hv, kwargs...)
end

"""
    OptimizationProblem(nlpmodel::AbstractNLPModel, adtype::AbstractADType = NoAD())

Returns an `OptimizationProblem` with the bounds and constraints defined in `nlpmodel`.
The optimization function and its derivatives are re-used from `nlpmodel` when available
or populated wit the Automatic Differentiation backend specified by `adtype`.
"""
function SciMLBase.OptimizationProblem(nlpmodel::AbstractNLPModel,
        adtype::ADTypes.AbstractADType = SciMLBase.NoAD(); kwargs...)
    f = OptimizationFunction(nlpmodel, adtype; kwargs...)
    u0 = nlpmodel.meta.x0
    lb, ub = if has_bounds(nlpmodel)
        (nlpmodel.meta.lvar, nlpmodel.meta.uvar)
    else
        (nothing, nothing)
    end

    lcons, ucons = if has_inequalities(nlpmodel) || has_equalities(nlpmodel)
        (nlpmodel.meta.lcon, nlpmodel.meta.ucon)
    else
        (nothing, nothing)
    end
    sense = nlpmodel.meta.minimize ? Optimization.MinSense : Optimization.MaxSense

    # The number of variables, geometry of u0, etc.. are valid and were checked when the
    # nlpmodel was created.

    return Optimization.OptimizationProblem(
        f, u0; lb = lb, ub = ub, lcons = lcons, ucons = ucons, sense = sense, kwargs...)
end

end
