using OptimizationIpopt, Aqua, JET
using Test

@testset "Aqua" begin
    Aqua.test_all(OptimizationIpopt)
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationIpopt; target_defined_modules = true)
end
