using SciMLTesting, OptimizationMultistartOptimization, JET
using Test

# ExplicitImports findings, all tracked against SciML/Optimization.jl:
#  * no_implicit_imports broken: the module relies on `@reexport`/`using`
#    module names (SciMLBase/OptimizationBase/Reexport/...) that cannot be made
#    explicit without restructuring.
#  * the ignored *_are_public / *_via_owners names are owned by SciMLBase,
#    OptimizationBase, the backend, or Base and are not (yet) declared public;
#    the proper fix is upstream `public` declarations, not a local change.
run_qa(
    OptimizationMultistartOptimization;
    explicit_imports = true,
    aqua_kwargs = (;
        # The sublibrary extends SciMLBase's solver-trait/__init/__solve interface
        # onto its backend's optimizer types, so those methods are intentional.
        piracies = (;
            treat_as_own = [
                OptimizationMultistartOptimization.MultistartOptimization.TikTak,
                OptimizationMultistartOptimization.OptimizationCache,
                OptimizationMultistartOptimization.SciMLBase.OptimizationProblem,
            ],
        ),
    ),
    ei_kwargs = (;
        all_qualified_accesses_via_owners = (; ignore = (:OptimizationStats,)),
        all_qualified_accesses_are_public = (; ignore = (:OptimizationStats, :__init, :__solve, :allowsbounds, :allowscallback, :build_solution, :has_init, :requiresbounds)),
    ),
    ei_broken = (:no_implicit_imports,),
)
