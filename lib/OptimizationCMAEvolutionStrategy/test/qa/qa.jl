using OptimizationCMAEvolutionStrategy, Aqua, JET
using Test

@testset "Aqua" begin
    Aqua.test_all(OptimizationCMAEvolutionStrategy)
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationCMAEvolutionStrategy; target_defined_modules = true)
end
