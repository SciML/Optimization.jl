using OptimizationManopt, Aqua, JET
using Test

@testset "Aqua" begin
    # `@reexport using Manopt` exports `solve!`, which clashes with SciMLBase's `solve!`
    # brought in transitively via OptimizationBase. The clash leaves OptimizationManopt's
    # `solve!` export pointing at neither binding; mark broken until the reexport is
    # restructured.
    # Manifolds is declared because the curvature analysis path may pull it in,
    # but no symbol from it is currently used in src — ignore it for now.
    Aqua.test_all(
        OptimizationManopt;
        undefined_exports = (; broken = true),
        stale_deps = (; ignore = [:Manifolds])
    )
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationManopt; target_defined_modules = true)
end
