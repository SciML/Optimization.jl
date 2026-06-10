using OptimizationPyCMA, Aqua, JET
using Test

@testset "Aqua" begin
    Aqua.test_all(OptimizationPyCMA)
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationPyCMA; target_defined_modules = true)
end
