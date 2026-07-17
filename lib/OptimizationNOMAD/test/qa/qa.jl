using SciMLTesting, OptimizationNOMAD, JET
using Test

include(normpath(joinpath(@__DIR__, "..", "..", "..", "..", "test", "qa", "rendered_docs.jl")))

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
# Conflicting export: `solve` (NOMAD + CommonSolve).
run_qa(
    OptimizationNOMAD;
    explicit_imports = true,
    aqua_broken = (:undefined_exports,),
    ei_kwargs = (;
        all_qualified_accesses_via_owners = (; ignore = (:OptimizationStats,)),
        all_qualified_accesses_are_public = (; ignore = (:DefaultOptimizationCache, :OptimizationStats, :__solve, :_check_and_convert_maxiters, :_check_and_convert_maxtime, :allowscallback)),
    ),
    api_docs_kwargs = (;
        rendered = true,
        docs_src = OPTIMIZATION_DOCS_SRC,
        rendered_ignore = optimization_dependency_rendered_ignore(OptimizationNOMAD),
        ignore = (
            :AutoModelingToolkit,
            :AutoSparseFastDifferentiation,
            :AutoSparseFiniteDiff,
            :AutoSparseForwardDiff,
            :AutoSparsePolyesterForwardDiff,
            :AutoSparseReverseDiff,
            :AutoSparseZygote,
            :solve,
        ),
    ),
    ei_broken = (:no_implicit_imports,),
)
