using SciMLTesting, OptimizationNLopt, JET
using Test

# ExplicitImports findings, all tracked against SciML/Optimization.jl:
#  * no_implicit_imports broken: the module relies on `@reexport`/`using`
#    module names (SciMLBase/OptimizationBase/Reexport/...) that cannot be made
#    explicit without restructuring.
#  * the ignored *_are_public / *_via_owners names are owned by SciMLBase,
#    OptimizationBase, the backend, or Base and are not (yet) declared public;
#    the proper fix is upstream `public` declarations, not a local change.
run_qa(
    OptimizationNLopt;
    explicit_imports = true,
    aqua_kwargs = (;
        # The sublibrary extends SciMLBase's solver-trait/__init/__solve interface
        # onto its backend's optimizer types, so those methods are intentional.
        piracies = (;
            treat_as_own = [
                OptimizationNLopt.NLopt.Algorithm,
                OptimizationNLopt.NLopt.Opt,
                OptimizationNLopt.OptimizationCache,
                OptimizationNLopt.SciMLBase.OptimizationProblem,
            ],
        ),
    ),
    ei_kwargs = (;
        all_qualified_accesses_via_owners = (; ignore = (:OptimizationStats,)),
        all_qualified_accesses_are_public = (; ignore = (:AUGLAG, :Failure, :LD_AUGLAG, :LN_AUGLAG, :OptimizationState, :OptimizationStats, :__init, :__solve, :_check_and_convert_maxiters, :_check_and_convert_maxtime, :allowsbounds, :allowscallback, :allowsconstraints, :build_solution, :has_init, :nlopt_set_param, :requiresconsjac, :requiresgradient, :requireshessian, :supports_sense)),
        all_explicit_imports_are_public = (; ignore = (:deduce_retcode,)),
    ),
    ei_broken = (:no_implicit_imports,),
)
