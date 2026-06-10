using OptimizationPolyalgorithms, Aqua, JET
using Test

@testset "Aqua" begin
    Aqua.test_all(OptimizationPolyalgorithms)
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationPolyalgorithms; target_defined_modules = true)
end
