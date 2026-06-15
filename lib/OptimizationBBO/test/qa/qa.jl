using OptimizationBBO, Aqua, JET
using Test

@testset "Aqua" begin
    Aqua.test_all(OptimizationBBO)
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationBBO; target_defined_modules = true)
end
