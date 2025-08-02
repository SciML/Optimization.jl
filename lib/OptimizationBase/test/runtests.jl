using OptimizationBase
using Test

@testset "OptimizationBase.jl" begin
    include("adtests.jl")
    include("cvxtest.jl")
    include("matrixvalued.jl")
end
