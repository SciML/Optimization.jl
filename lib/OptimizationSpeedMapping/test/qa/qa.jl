using OptimizationSpeedMapping, Aqua, JET
using Test

@testset "Aqua" begin
    Aqua.test_all(OptimizationSpeedMapping)
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationSpeedMapping; target_defined_modules = true)
end
