module OptimizationMooncakeExt

using OptimizationBase, Mooncake
using SciMLBase
using Mooncake: rrule!!, CoDual, zero_fcodual, @is_primitive,
    @from_chainrules, @zero_adjoint, @mooncake_overlay, MinimalCtx,
    NoPullback

@from_chainrules MinimalCtx Tuple{
    typeof(OptimizationBase.solve_up),
    SciMLBase.AbstractOptimizationProblem,
    Union{Nothing, SciMLBase.AbstractSensitivityAlgorithm},
    Any,
    Any,
    Any,
} true

# Dispatch for auto-alg (no explicit algorithm in args)
@from_chainrules MinimalCtx Tuple{
    typeof(OptimizationBase.solve_up),
    SciMLBase.AbstractOptimizationProblem,
    Union{Nothing, SciMLBase.AbstractSensitivityAlgorithm},
    Any,
    Any,
} true

end