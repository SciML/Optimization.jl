using SimpleOptimization, Aqua, JET
using Test

@testset "Aqua" begin
    Aqua.test_all(SimpleOptimization)
end

@testset "JET static analysis" begin
    JET.test_package(SimpleOptimization; target_defined_modules = true)
end
