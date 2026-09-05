using Test

@testset "OptimizationBase AD" begin
    include("enzyme_lagrangian_hessian.jl")
    include("adtests.jl")
    include("dual_tolerant_tests.jl")
    include("cvxtest.jl")
end
