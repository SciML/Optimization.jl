using SciMLTesting, OptimizationEvolutionary, JET
using Test

# ExplicitImports findings, all tracked against SciML/Optimization.jl:
#  * no_implicit_imports broken: the module relies on `@reexport`/`using`
#    module names (SciMLBase/OptimizationBase/Reexport/...) that cannot be made
#    explicit without restructuring.
#  * the ignored *_are_public / *_via_owners names are owned by SciMLBase,
#    OptimizationBase, the backend, or Base and are not (yet) declared public;
#    the proper fix is upstream `public` declarations, not a local change.
run_qa(
    OptimizationEvolutionary;
    explicit_imports = true,
    aqua_kwargs = (;
        # The sublibrary extends SciMLBase's solver-trait/__init/__solve interface
        # onto its backend's optimizer types, so those methods are intentional.
        piracies = (;
            treat_as_own = [
                OptimizationEvolutionary.Evolutionary.AbstractOptimizer,
                OptimizationEvolutionary.Evolutionary.NSGA2,
                OptimizationEvolutionary.OptimizationCache,
            ],
        ),
    ),
    ei_kwargs = (;
        all_qualified_accesses_via_owners = (; ignore = (:OptimizationStats, :minimum)),
        all_qualified_accesses_are_public = (; ignore = (:AbstractOptimizer, :OptimizationState, :OptimizationStats, :OptimizationTrace, :OptimizationTraceRecord, :Options, :__solve, :_check_and_convert_maxiters, :_check_and_convert_maxtime, :allowscallback, :converged, :minimizer, :minimum, :optimize, :requiresconshess, :requiresconsjac, :requiresgradient, :requireshessian, :trace!, :update!)),
    ),
    ei_broken = (:no_implicit_imports,),
)
