using SciMLTesting, OptimizationNLPModels, JET
using Test
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
SB = OptimizationNLPModels.SciMLBase
include(normpath(joinpath(@__DIR__, "..", "..", "..", "..", "test", "qa", "public_api_docs.jl")))

run_qa(
    OptimizationNLPModels;
    explicit_imports = true,
    aqua_kwargs = (;
        piracies = (;
            treat_as_own = [SB.OptimizationFunction, SB.OptimizationProblem],
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
    api_docs_kwargs = public_api_docs_kwargs(OptimizationNLPModels),
    ei_broken = (:no_implicit_imports,),
)
