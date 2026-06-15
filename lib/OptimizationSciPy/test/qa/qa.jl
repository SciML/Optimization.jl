using OptimizationSciPy, Aqua, JET
using Test

@testset "Aqua" begin
    Aqua.test_all(OptimizationSciPy)
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationSciPy; target_defined_modules = true)
end
