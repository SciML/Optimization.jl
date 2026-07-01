using OptimizationManopt, Aqua, JET
using Test

@testset "Aqua" begin
    # Manifolds is declared because the curvature analysis path may pull it in,
    # but no symbol from it is currently used in src — ignore it for now.
    Aqua.test_all(
        OptimizationManopt;
        stale_deps = (; ignore = [:Manifolds])
    )
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationManopt; target_defined_modules = true)
end
