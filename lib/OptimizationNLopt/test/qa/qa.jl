using OptimizationNLopt, Aqua, JET
using Test

@testset "Aqua" begin
    Aqua.test_all(OptimizationNLopt)
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationNLopt; target_defined_modules = true)
end
