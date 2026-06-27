using SciMLTesting, OptimizationMultistartOptimization, JET
using Test
using MultistartOptimization

# ExplicitImports findings, all tracked against SciML/Optimization.jl:
#  * no_implicit_imports broken: the module relies on `@reexport`/`using`
#    module names (SciMLBase/OptimizationBase/Reexport/...) that cannot be made
#    explicit without restructuring.
#  * the ignored *_are_public / *_via_owners names are owned by SciMLBase,
#    OptimizationBase, the backend, or Base and are not (yet) declared public;
#    the proper fix is upstream `public` declarations, not a local change.
# OptimizationMultistartOptimization implements the SciML optimization interface
# for MultistartOptimization, so the trait/interface methods it adds extend
# SciML's *own* functions rather than committing type piracy — mark those
# functions as own.
SB = OptimizationMultistartOptimization.SciMLBase
run_qa(
    OptimizationMultistartOptimization;
    explicit_imports = true,
    aqua_kwargs = (;
        piracies = (;
            treat_as_own = [
                SB.__init,
                SB.__solve,
                SB.allowsbounds,
                SB.allowscallback,
                SB.has_init,
                SB.requiresbounds,
            ],
        ),
        # OptimizationNLopt is used in tests as the inner solver, not in src.
        stale_deps = (; ignore = [:OptimizationNLopt]),
    ),
    ei_kwargs = (;
        all_qualified_accesses_via_owners = (; ignore = (:OptimizationStats,)),
        all_qualified_accesses_are_public = (; ignore = (:OptimizationStats, :__init, :__solve, :allowscallback, :requiresbounds)),
    ),
    ei_broken = (:no_implicit_imports,),
)
