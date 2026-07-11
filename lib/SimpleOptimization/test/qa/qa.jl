using SciMLTesting, SimpleOptimization, JET
using Test

# ExplicitImports findings, all tracked against SciML/Optimization.jl:
#  * no_implicit_imports broken: the module relies on `@reexport`/`using`
#    module names (SciMLBase/OptimizationBase/Reexport/...) that cannot be made
#    explicit without restructuring.
#  * the ignored *_are_public / *_via_owners names are owned by SciMLBase,
#    OptimizationBase, the backend, or Base and are not (yet) declared public;
#    the proper fix is upstream `public` declarations, not a local change.
include(normpath(joinpath(@__DIR__, "..", "..", "..", "..", "test", "qa", "public_api_docs.jl")))

run_qa(
    SimpleOptimization;
    explicit_imports = true,
    ei_kwargs = (;
        all_qualified_accesses_via_owners = (; ignore = (:OptimizationStats,)),
        # `gradient` (ForwardDiff) and `instantiate_gradient` (SimpleOptimization's
        # own internal) are non-public in their owners; `_unwrap_val` is a SciMLBase
        # internal still non-public on the registered release (SciMLBase 3.24.0).
        all_qualified_accesses_are_public = (; ignore = (:gradient, :instantiate_gradient, :OptimizationStats, :__solve, :_check_and_convert_maxiters)),
        all_explicit_imports_are_public = (; ignore = (:_unwrap_val,)),
    ),
    api_docs_kwargs = public_api_docs_kwargs(SimpleOptimization),
    ei_broken = (:no_implicit_imports,),
)
