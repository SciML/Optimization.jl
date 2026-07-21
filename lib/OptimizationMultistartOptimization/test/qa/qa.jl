using SciMLTesting, OptimizationMultistartOptimization, JET, SciMLBase
using Test

include(normpath(joinpath(@__DIR__, "..", "..", "..", "..", "test", "qa", "rendered_docs.jl")))

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
run_qa(
    OptimizationMultistartOptimization;
    explicit_imports = true,
    aqua_kwargs = (;
        piracies = (;
            treat_as_own = [
                SciMLBase.__init,
                SciMLBase.__solve,
                SciMLBase.allowsbounds,
                SciMLBase.allowscallback,
                SciMLBase.has_init,
                SciMLBase.requiresbounds,
            ],
        ),
        # OptimizationNLopt is used in tests as the inner solver, not in src.
        stale_deps = (; ignore = [:OptimizationNLopt]),
    ),
    ei_kwargs = (;
        all_qualified_accesses_via_owners = (; ignore = (:OptimizationStats,)),
        all_qualified_accesses_are_public = (; ignore = (:OptimizationStats, :__init, :__solve, :allowscallback, :requiresbounds)),
    ),
    api_docs_kwargs = (;
        rendered = true,
        docs_src = OPTIMIZATION_DOCS_SRC,
        rendered_ignore = optimization_dependency_rendered_ignore(OptimizationMultistartOptimization),
        ignore = (
            :AutoModelingToolkit,
            :AutoSparseFastDifferentiation,
            :AutoSparseFiniteDiff,
            :AutoSparseForwardDiff,
            :AutoSparsePolyesterForwardDiff,
            :AutoSparseReverseDiff,
            :AutoSparseZygote,
        ),
    ),
    ei_broken = (:no_implicit_imports,),
)
