using OptimizationMadNLP, Aqua, JET
using Test

@testset "Aqua" begin
    Aqua.test_all(OptimizationMadNLP)
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationMadNLP; target_defined_modules = true)
end
