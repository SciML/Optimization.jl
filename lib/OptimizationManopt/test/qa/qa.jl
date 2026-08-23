using SciMLTesting, OptimizationManopt, JET
using Test

include(normpath(joinpath(@__DIR__, "..", "..", "..", "..", "test", "qa", "rendered_docs.jl")))

# ExplicitImports findings, all tracked against SciML/Optimization.jl:
#  * no_implicit_imports broken: the module relies on `using` module names
#    (SciMLBase/OptimizationBase/...) that cannot be made
#    explicit without restructuring.
#  * the ignored *_are_public / *_via_owners names are owned by SciMLBase,
#    OptimizationBase, the backend, or Base and are not (yet) declared public;
#    the proper fix is upstream `public` declarations, not a local change.
run_qa(
    OptimizationManopt;
    explicit_imports = true,
    aqua_kwargs = (;
        # Manifolds is declared because the curvature analysis path may pull it in,
        # but no symbol from it is currently used in src — ignore it for now.
        stale_deps = (; ignore = [:Manifolds]),
    ),
    ei_kwargs = (;
        all_qualified_accesses_via_owners = (; ignore = (:AbstractManifold,)),
        all_qualified_accesses_are_public = (; ignore = (:AbstractManifold, :OptimizationState, :__solve, :allowscallback, :build_solution, :requiresgradient, :requireshessian, :supports_sense)),
    ),
    ei_broken = (:no_implicit_imports,),
    # Only the `Manopt` module binding is re-surfaced; every Manopt name the docs use is
    # spelled qualified through it. Kept in sync with src/OptimizationManopt.jl.
    reexports_allow = (:Manopt,),
)

@testset "reexported Manopt module binding" begin
    test_reexported_names(OptimizationManopt, (:Manopt,))
end
