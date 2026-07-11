using SciMLTesting, OptimizationSciPy, JET
using Test

# ExplicitImports findings, all tracked against SciML/Optimization.jl. This env
# can only be analyzed where CondaPkg can run; the broken checks are the
# `@reexport`/`using` module-name relies plus qualified accesses to SciMLBase/
# OptimizationBase internals that are not (yet) declared public.
include(normpath(joinpath(@__DIR__, "..", "..", "..", "..", "test", "qa", "public_api_docs.jl")))

run_qa(
    OptimizationSciPy;
    explicit_imports = true,
    api_docs_kwargs = public_api_docs_kwargs(OptimizationSciPy),
    ei_broken = (
        :no_implicit_imports,
        :all_qualified_accesses_via_owners,
        :all_qualified_accesses_are_public,
    ),
)
