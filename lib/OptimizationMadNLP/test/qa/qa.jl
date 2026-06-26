using SciMLTesting, OptimizationMadNLP, JET
using Test

# ExplicitImports findings, all tracked against SciML/Optimization.jl:
#  * no_implicit_imports broken: the module relies on `@reexport`/`using`
#    module names (SciMLBase/OptimizationBase/Reexport/...) that cannot be made
#    explicit without restructuring.
#  * the ignored *_are_public / *_via_owners names are owned by SciMLBase,
#    OptimizationBase, the backend, or Base and are not (yet) declared public;
#    the proper fix is upstream `public` declarations, not a local change.
# undefined_exports broken (tracked against SciML/Optimization.jl): the module
# `@reexport`s two packages that both export the name below, leaving it an
# unresolved conflict binding; the fix is a maintainer choice of which wins.
# Conflicting export: `solve!` (MadNLP + OptimizationBase).
run_qa(
    OptimizationMadNLP;
    explicit_imports = true,
    aqua_broken = (:undefined_exports,),
    ei_kwargs = (;
        all_qualified_accesses_via_owners = (; ignore = (:OptimizationStats,)),
        all_qualified_accesses_are_public = (; ignore = (Symbol("@logmsg"), :AbstractBarrierUpdate, :AbstractMadNLPSolver, :AbstractUserCallback, :DIVERGING_ITERATES, :ExactHessian, :INFEASIBLE_PROBLEM_DETECTED, :INFO, :LogLevel, :LogLevels, :MAXIMUM_ITERATIONS_EXCEEDED, :MAXIMUM_WALLTIME_EXCEEDED, :NOT_ENOUGH_DEGREES_OF_FREEDOM, :OptimizationState, :OptimizationStats, :QuasiNewtonOptions, :RESTORATION_FAILED, :SEARCH_DIRECTION_BECOMES_TOO_SMALL, :SOLVED_TO_ACCEPTABLE_LEVEL, :SOLVE_SUCCEEDED, :Status, :USER_REQUESTED_STOP, :WARN, :__solve, :_check_and_convert_maxiters, :_check_and_convert_maxtime, :allowsbounds, :allowscallback, :allowsconsjvp, :allowsconstraints, :allowsconsvjp, :requiresconshess, :requiresconsjac, :requiresgradient, :requireshessian, :requireslagh, :supports_sense)),
    ),
    ei_broken = (:no_implicit_imports,),
)
