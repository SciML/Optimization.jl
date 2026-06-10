using OptimizationMultistartOptimization, Aqua, JET
using Test

@testset "Aqua" begin
    Aqua.test_all(OptimizationMultistartOptimization)
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationMultistartOptimization; target_defined_modules = true)
end
