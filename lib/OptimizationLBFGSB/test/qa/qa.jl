using OptimizationLBFGSB, Aqua, JET
using Test

@testset "Aqua" begin
    Aqua.test_all(OptimizationLBFGSB)
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationLBFGSB; target_defined_modules = true)
end
