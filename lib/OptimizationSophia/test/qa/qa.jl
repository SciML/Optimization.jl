using OptimizationSophia, Aqua, JET
using Test

@testset "Aqua" begin
    Aqua.test_all(OptimizationSophia)
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationSophia; target_defined_modules = true)
end
