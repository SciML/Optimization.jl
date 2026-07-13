using Test

@testset "OptimizationBase AD" begin
    include("adtests.jl")
    include("dual_tolerant_tests.jl")
    include("cvxtest.jl")
end
