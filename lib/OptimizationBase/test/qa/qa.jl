using OptimizationBase, Aqua, JET
using Test

@testset "Aqua" begin
    Aqua.test_all(OptimizationBase)
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationBase; target_defined_modules = true)
end
