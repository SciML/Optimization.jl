using OptimizationMOI, Aqua, JET
using Test

@testset "Aqua" begin
    Aqua.test_all(OptimizationMOI)
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationMOI; target_defined_modules = true)
end
