using SciMLTesting, OptimizationNLPModels, JET, SciMLBase
using Test

include(normpath(joinpath(@__DIR__, "..", "..", "..", "..", "test", "qa", "rendered_docs.jl")))

using NLPModels

# ExplicitImports findings, all tracked against SciML/Optimization.jl:
#  * no_implicit_imports broken: the module relies on `@reexport`/`using`
#    module names (SciMLBase/OptimizationBase/Reexport/...) that cannot be made
#    explicit without restructuring.
#  * the ignored *_are_public / *_via_owners names are owned by SciMLBase,
#    OptimizationBase, the backend, or Base and are not (yet) declared public;
#    the proper fix is upstream `public` declarations, not a local change.
# This package defines SciMLBase.OptimizationFunction / SciMLBase.OptimizationProblem
# constructors for NLPModels.AbstractNLPModel. Those constructors extend SciML's
# *own* types, so mark those (not the NLPModels type) as own for the piracy check.
run_qa(
    OptimizationNLPModels;
    explicit_imports = true,
    aqua_kwargs = (;
        piracies = (;
            treat_as_own = [SciMLBase.OptimizationFunction, SciMLBase.OptimizationProblem],
        ),
        # Optimization sublibraries used to construct OptimizationProblems in tests
        # but not `using`d in src; Aqua's static analysis can't see test-only usage.
        stale_deps = (; ignore = [:OptimizationLBFGSB, :OptimizationMOI, :OptimizationOptimJL]),
    ),
    ei_kwargs = (;
        # `NoAD` is owned by SciMLBase and remains non-public there on the registered
        # release (SciMLBase 3.24.0); fix belongs upstream via a `public` declaration.
        all_qualified_accesses_are_public = (; ignore = (:NoAD,)),
    ),
    api_docs_kwargs = (;
        rendered = true,
        docs_src = OPTIMIZATION_DOCS_SRC,
        rendered_ignore = optimization_dependency_rendered_ignore(OptimizationNLPModels),
        ignore = (
            :AutoModelingToolkit,
            :AutoSparseFastDifferentiation,
            :AutoSparseFiniteDiff,
            :AutoSparseForwardDiff,
            :AutoSparsePolyesterForwardDiff,
            :AutoSparseReverseDiff,
            :AutoSparseZygote,
            :get_nbatch,
            :jth_con,
            :jth_congrad,
            :jth_congrad!,
            :jth_sparse_congrad,
        ),
    ),
    ei_broken = (:no_implicit_imports,),
)
