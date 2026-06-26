using SciMLTesting, OptimizationAuglag, JET
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
    OptimizationAuglag;
    explicit_imports = true,
    aqua_kwargs = (;
        # OptimizationOptimisers is used in tests as the inner solver, not in src.
        stale_deps = (; ignore = [:OptimizationOptimisers]),
    ),
    ei_kwargs = (;
        no_stale_explicit_imports = (; ignore = (:norm,)),
        all_qualified_accesses_are_public = (; ignore = (:OptimizationState, :__solve, :_check_and_convert_maxiters, :allowsbounds, :allowscallback, :allowsconstraints, :allowsfg, :isa_dataiterator, :requiresconsjac, :requiresgradient)),
        all_explicit_imports_are_public = (; ignore = (:OptimizationStats,)),
    ),
    ei_broken = (:no_implicit_imports,),
)
