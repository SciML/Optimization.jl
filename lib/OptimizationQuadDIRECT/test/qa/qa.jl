using OptimizationQuadDIRECT, Aqua, JET
using Test

@testset "Aqua" begin
    Aqua.test_all(OptimizationQuadDIRECT)
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationQuadDIRECT; target_defined_modules = true)
end
