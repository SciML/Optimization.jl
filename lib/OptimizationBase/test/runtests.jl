using OptimizationBase
using Test

@testset "OptimizationBase.jl" begin
    include("adtests.jl")
    include("cvxtest.jl")
    include("matrixvalued.jl")
    include("solver_missing_error_messages.jl")
end
