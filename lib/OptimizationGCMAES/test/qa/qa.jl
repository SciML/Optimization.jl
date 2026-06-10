using OptimizationGCMAES, Aqua, JET
using Test

@testset "Aqua" begin
    Aqua.test_all(OptimizationGCMAES)
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationGCMAES; target_defined_modules = true)
end
