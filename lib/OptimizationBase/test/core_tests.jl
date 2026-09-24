using OptimizationBase
using Test

@testset "OptimizationBase.jl" begin
    include("matrixvalued.jl")
    include("solver_missing_error_messages.jl")
    include("lag_h_sigma_zero_test.jl")
    include("second_order_warning_test.jl")
    include("solve_internals_test.jl")
end
