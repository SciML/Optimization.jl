using OptimizationNOMAD, Aqua, JET
using Test

@testset "Aqua" begin
    Aqua.test_all(OptimizationNOMAD)
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationNOMAD; target_defined_modules = true)
end
