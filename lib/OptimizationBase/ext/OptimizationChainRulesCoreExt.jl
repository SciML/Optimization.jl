module OptimizationChainRulesCoreExt

using OptimizationBase
using SciMLBase
using SciMLBase: AbstractSensitivityAlgorithm, AbstractOptimizationProblem

import ChainRulesCore
import ChainRulesCore: NoTangent, Tangent

function ChainRulesCore.frule(
        ::typeof(OptimizationBase.solve_up), prob,
        sensealg::Union{Nothing, AbstractSensitivityAlgorithm},
        u0, p, args...; originator = SciMLBase.ChainRulesOriginator(),
        kwargs...
    )
    return OptimizationBase._solve_forward(
        prob, sensealg, u0, p,
        originator, args...;
        kwargs...
    )
end

function ChainRulesCore.rrule(
        ::typeof(OptimizationBase.solve_up), prob::AbstractOptimizationProblem,
        sensealg::Union{Nothing, AbstractSensitivityAlgorithm},
        u0, p, args...; originator = SciMLBase.ChainRulesOriginator(),
        kwargs...
    )
    primal, inner_thunking_pb = OptimizationBase._solve_adjoint(
        prob, sensealg, u0, p,
        originator, args...;
        kwargs...
    )

    function solve_up_adjoint(∂sol)
        return inner_thunking_pb(∂sol isa Tangent{Any, <:NamedTuple} ? ∂sol.u : ∂sol)
    end
    return primal, solve_up_adjoint
end

end