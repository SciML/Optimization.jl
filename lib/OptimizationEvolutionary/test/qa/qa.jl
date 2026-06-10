using OptimizationEvolutionary, Aqua, JET
using Test

@testset "Aqua" begin
    Aqua.test_all(OptimizationEvolutionary)
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationEvolutionary; target_defined_modules = true)
end
