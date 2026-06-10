using OptimizationManopt, Aqua, JET
using Test

@testset "Aqua" begin
    Aqua.test_all(OptimizationManopt)
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationManopt; target_defined_modules = true)
end
