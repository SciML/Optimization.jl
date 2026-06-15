using OptimizationOptimisers, Aqua, JET
using Test

@testset "Aqua" begin
    Aqua.test_all(OptimizationOptimisers)
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationOptimisers; target_defined_modules = true)
end
