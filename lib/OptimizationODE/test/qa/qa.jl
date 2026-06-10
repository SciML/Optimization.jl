using OptimizationODE, Aqua, JET
using Test

@testset "Aqua" begin
    Aqua.test_all(OptimizationODE)
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationODE; target_defined_modules = true)
end
