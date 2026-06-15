using OptimizationMetaheuristics, Aqua, JET
using Test

@testset "Aqua" begin
    Aqua.test_all(OptimizationMetaheuristics)
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationMetaheuristics; target_defined_modules = true)
end
