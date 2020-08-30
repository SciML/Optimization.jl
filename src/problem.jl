abstract type AbstractOptimizationProblem end

struct OptimizationProblem{F,X,P,B,LC,UC,K} <: AbstractOptimizationProblem
    f::F
    x::X
    p::P
    lb::B
    ub::B
    lcons::LC
    ucons::UC
    kwargs::K
    function OptimizationProblem(f, x; p=DiffEqBase.NullParameters(), lb = [], ub = [], lcons = [], ucons = [], kwargs...)
        new{typeof(f), typeof(x), typeof(p), typeof(lb), typeof(lcons), typeof(ucons), typeof(kwargs)}(f, x, p, lb, ub, lcons, ucons, kwargs)
    end
end

import MathOptInterface
const MOI = MathOptInterface

struct MOIOptimizationProblem <: MOI.AbstractNLPEvaluator
    prob::OptimizationProblem
end

MOI.eval_objective(moiproblem::MOIOptimizationProblem, x) = moiproblem.prob.f isa OptimizationFunction ? moiproblem.prob.f.f(x) : moiproblem.prob.f(x)

MOI.eval_objective_gradient(moiproblem::MOIOptimizationProblem, G, x) = moiproblem.prob.f isa OptimizationFunction ? moiproblem.prob.f.grad(G, x) : error("Use OptimizationFunction to pass in gradient function")

MOI.eval_hessian_lagrangian(moiproblem::MOIOptimizationProblem, H, x, σ, μ) = moiproblem.prob.f isa OptimizationFunction ? σ.* moiproblem.prob.f.hess(H, x) : error("Use OptimizationFunction to pass in hessian function")

function MOI.initialize(moiproblem::MOIOptimizationProblem, requested_features::Vector{Symbol})
    for feat in requested_features
        if !(feat in MOI.features_available(moiproblem))
            error("Unsupported feature $feat")
            # TODO: implement Jac-vec and Hess-vec products
            # for solvers that need them
        end
    end
end

function MOI.features_available(moiproblem::MOIOptimizationProblem)
    if moiproblem.prob.f isa OptimizationFunction
        return [:Grad, :Hess]
    else
        error("Use OptimizationFunction to pass in gradient and hessian")
    end
end

function make_moi_problem(prob::OptimizationProblem)
    moiproblem = MOIOptimizationProblem(prob)
    return moiproblem
end