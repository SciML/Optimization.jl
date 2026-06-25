using SciMLTesting, OptimizationOptimJL, JET
using Test

# ExplicitImports findings, all tracked against SciML/Optimization.jl:
#  * no_implicit_imports broken: the module relies on `@reexport`/`using`
#    module names (SciMLBase/OptimizationBase/Reexport/...) that cannot be made
#    explicit without restructuring.
#  * the ignored *_are_public / *_via_owners names are owned by SciMLBase,
#    OptimizationBase, the backend, or Base and are not (yet) declared public;
#    the proper fix is upstream `public` declarations, not a local change.
run_qa(
    OptimizationOptimJL;
    explicit_imports = true,
    aqua_kwargs = (;
        # The sublibrary extends SciMLBase's solver-trait/__init/__solve interface
        # onto its backend's optimizer types, so those methods are intentional.
        piracies = (;
            treat_as_own = [
                OptimizationOptimJL.Optim.AbstractOptimizer,
                OptimizationOptimJL.Optim.ConstrainedOptimizer,
                OptimizationOptimJL.Optim.Fminbox,
                OptimizationOptimJL.Optim.IPNewton,
                OptimizationOptimJL.Optim.KrylovTrustRegion,
                OptimizationOptimJL.Optim.Newton,
                OptimizationOptimJL.Optim.NewtonTrustRegion,
                OptimizationOptimJL.Optim.SAMIN,
                OptimizationOptimJL.Optim.SimulatedAnnealing,
                OptimizationOptimJL.OptimizationCache,
                OptimizationOptimJL.SciMLBase.OptimizationProblem,
            ],
        ),
    ),
    ei_kwargs = (;
        all_qualified_accesses_via_owners = (; ignore = (:NLSolversBase, :OptimizationStats)),
        all_qualified_accesses_are_public = (; ignore = (:AbstractConstrainedOptimizer, :AbstractOptimizer, :ConstrainedOptimizer, :Failure, :KrylovTrustRegion, :MaxIters, :NLSolversBase, :NoAD, :OptimizationState, :OptimizationStats, :Options, :Success, :ZerothOrderOptimizer, :__init, :__solve, :_check_and_convert_maxiters, :_check_and_convert_maxtime, :alloc_DF, :alloc_H, :allowsbounds, :allowscallback, :allowsconstraints, :allowsfg, :build_solution, :converged, :has_init, :iteration_limit_reached, :requiresbounds, :requiresconshess, :requiresconsjac, :requiresgradient, :requireshessian, :supports_sense, :value)),
    ),
    ei_broken = (:no_implicit_imports,),
)
