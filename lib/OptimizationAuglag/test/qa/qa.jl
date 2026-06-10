using OptimizationAuglag, Aqua, JET
using Test

@testset "Aqua" begin
    Aqua.test_all(OptimizationAuglag)
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationAuglag; target_defined_modules = true)
end
