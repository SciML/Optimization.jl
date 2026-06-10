using OptimizationNLPModels, Aqua, JET
using Test

@testset "Aqua" begin
    Aqua.test_all(OptimizationNLPModels)
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationNLPModels; target_defined_modules = true)
end
