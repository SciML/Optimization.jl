using OptimizationAuglag, Aqua, JET
using Test

@testset "Aqua" begin
    # OptimizationOptimisers is used in tests as the inner solver, not in src.
    Aqua.test_all(
        OptimizationAuglag;
        stale_deps = (; ignore = [:OptimizationOptimisers])
    )
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationAuglag; target_defined_modules = true)
end
