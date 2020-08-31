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

MOI.eval_constraint(moiproblem::MOIOptimizationProblem, g, x) = moiproblem.prob.f isa OptimizationFunction ? g .= moiproblem.prob.f.cons(x) : error("Use OptimizationFunction to pass in constraints function")

MOI.eval_objective_gradient(moiproblem::MOIOptimizationProblem, G, x) = moiproblem.prob.f isa OptimizationFunction ? moiproblem.prob.f.grad(G, x) : error("Use OptimizationFunction to pass in gradient function")

MOI.eval_hessian_lagrangian(moiproblem::MOIOptimizationProblem, H, x, σ, μ) = moiproblem.prob.f isa OptimizationFunction ? σ.* moiproblem.prob.f.hess(H, x) + μ .*moiproblem.prob.f.cons_h(H,x) : error("Use OptimizationFunction to pass in hessian function")

MOI.eval_constraint_jacobian(moiproblem::MOIOptimizationProblem, J, x) = moiproblem.prob.f isa OptimizationFunction ? moiproblem.prob.f.cons_j(J, x) : error("Use OptimizationFunction to pass in jacobian function")

function MOI.jacobian_structure(moiproblem::MOIOptimizationProblem)
    return Tuple{Int64,Int64}[(1,i) for i in 1:length(moiproblem.prob.x)]
end

function MOI.hessian_lagrangian_structure(moiproblem::MOIOptimizationProblem)
    return vcat([[(j,i) for i in 1:length(moiproblem.prob.x)] for j in 1:length(moiproblem.prob.x)]...)
end

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
        return [:Grad, :Hess, :Jac]
    else
        error("Use OptimizationFunction to pass in gradient and hessian")
    end
end

function make_moi_problem(prob::OptimizationProblem)
    moiproblem = MOIOptimizationProblem(prob)
    return moiproblem
end