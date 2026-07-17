using SciMLTesting, OptimizationODE, JET
using Test

include(normpath(joinpath(@__DIR__, "..", "..", "..", "..", "test", "qa", "rendered_docs.jl")))

# ExplicitImports findings, all tracked against SciML/Optimization.jl:
#  * no_implicit_imports broken: the module relies on `@reexport`/`using`
#    module names (SciMLBase/OptimizationBase/Reexport/...) that cannot be made
#    explicit without restructuring.
#  * the ignored *_are_public / *_via_owners names are owned by SciMLBase,
#    OptimizationBase, the backend, or Base and are not (yet) declared public;
#    the proper fix is upstream `public` declarations, not a local change.
run_qa(
    OptimizationODE;
    explicit_imports = true,
    ei_kwargs = (;
        all_qualified_accesses_via_owners = (; ignore = (:OptimizationStats,)),
        all_qualified_accesses_are_public = (; ignore = (:OptimizationStats, :__init, :__solve, :allowscallback, :requiresbounds, :requiresconshess, :requiresconsjac, :requiresgradient, :requireshessian)),
    ),
    api_docs_kwargs = (;
        rendered = true,
        docs_src = OPTIMIZATION_DOCS_SRC,
        rendered_ignore = optimization_dependency_rendered_ignore(OptimizationODE),
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
