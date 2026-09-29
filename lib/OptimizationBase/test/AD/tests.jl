using SafeTestsets, Test

@testset "OptimizationBase AD" begin
    include("enzyme_lagrangian_hessian.jl")
    @safetestset "despecialize: Enzyme" include("despecialize_enzyme.jl")
    include("adtests.jl")
    include("dual_tolerant_tests.jl")
    include("cvxtest.jl")
end
