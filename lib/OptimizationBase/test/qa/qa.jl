using SciMLTesting, OptimizationBase, JET
using Test

# ExplicitImports findings, all tracked against SciML/Optimization.jl:
#  * no_implicit_imports broken: the module relies on `@reexport`/`using`
#    module names (SciMLBase/OptimizationBase/Reexport/...) that cannot be made
#    explicit without restructuring.
#  * the ignored *_are_public / *_via_owners names are owned by SciMLBase,
#    OptimizationBase, the backend, or Base and are not (yet) declared public;
#    the proper fix is upstream `public` declarations, not a local change.
#  * the ignored stale imports are part of the intentionally re-surfaced API.
run_qa(
    OptimizationBase;
    explicit_imports = true,
    aqua_kwargs = (;
        # The sublibrary extends SciMLBase's solver-trait/__init/__solve interface
        # onto its backend's optimizer types, so those methods are intentional.
        piracies = (;
            treat_as_own = [
                OptimizationBase.SciMLBase.OptimizationProblem,
                OptimizationBase.SciMLBase.AbstractOptimizationCache,
            ],
        ),
    ),
    ei_kwargs = (;
        no_stale_explicit_imports = (; ignore = (:I, :OptimizationStats, :extract_alg)),
        all_qualified_accesses_via_owners = (; ignore = (:IsInfinite, :IteratorSize, :SizeUnknown)),
        # Names imported/accessed from SciMLBase (plus Base.Iterators and
        # SparseConnectivityTracer's AbstractTracer) that remain non-public in their
        # source pkg on the registered releases (SciMLBase 3.27.0). The fix belongs
        # upstream via `public` declarations there, not a local change.
        all_qualified_accesses_are_public = (; ignore = (:AbstractOptimizationCache, :AbstractOptimizationFunction, :AbstractOptimizationSolution, :AbstractTracer, :ChainRulesOriginator, :IsInfinite, :IteratorSize, :MaxSense, :MinSense, :NoAD, :NonConcreteEltypeError, :SizeUnknown, :__init, :allowsconsjvp, :allowsconsvjp, :allowsfg, :allowsfgh, :requiresconshess, :requiresconsjac, :requiresgradient, :requireshessian, :requireslagh)),
        all_explicit_imports_are_public = (; ignore = (:KeywordArgError, :MaxSense, :MinSense, :ObjSense, :OptimizationStats, :__init, :__solve, :_concrete_solve_adjoint, :_concrete_solve_forward, :allowscallback, :extract_alg, :get_concrete_p, :get_concrete_u0, :get_root_indp, :get_updated_symbolic_problem, :has_kwargs, :promote_u0, :requiresbounds, :requiresconshess, :requiresconsjac, :requiresconstraints, :requiresgradient, :requireshessian, :wrap_sol)),
    ),
    ei_broken = (:no_implicit_imports,),
)
