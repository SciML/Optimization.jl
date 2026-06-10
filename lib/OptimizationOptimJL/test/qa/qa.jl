using OptimizationOptimJL, Aqua, JET
using Test

@testset "Aqua" begin
    Aqua.test_all(OptimizationOptimJL)
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationOptimJL; target_defined_modules = true)
end
