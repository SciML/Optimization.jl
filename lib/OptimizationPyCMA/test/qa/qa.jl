using SciMLTesting, OptimizationPyCMA, JET
using Test

include(normpath(joinpath(@__DIR__, "..", "..", "..", "..", "test", "qa", "rendered_docs.jl")))

# ExplicitImports findings, all tracked against SciML/Optimization.jl. This env
# can only be analyzed where CondaPkg can run; the broken checks are the
# `@reexport`/`using` module-name relies plus qualified accesses to SciMLBase/
# OptimizationBase internals that are not (yet) declared public.
run_qa(
    OptimizationPyCMA;
    explicit_imports = true,
    api_docs_kwargs = (;
        rendered = true,
        docs_src = OPTIMIZATION_DOCS_SRC,
        rendered_ignore = optimization_dependency_rendered_ignore(OptimizationPyCMA),
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
    ei_broken = (
        :no_implicit_imports,
        :all_qualified_accesses_via_owners,
        :all_qualified_accesses_are_public,
    ),
)
