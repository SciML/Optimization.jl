using Optimization
using ForwardDiff, Zygote, ReverseDiff, FiniteDiff
using Test

x0 = zeros(2)
rosenbrock(x, p = nothing) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
l1 = rosenbrock(x0)

optf = OptimizationFunction(rosenbrock, AutoForwardDiff())
prob = OptimizationProblem(optf, x0)
@time res = solve(prob, Optimization.LBFGS(), maxiters = 100)
@test res.retcode == Optimization.SciMLBase.ReturnCode.Success

prob = OptimizationProblem(optf, x0, lb = [-1.0, -1.0], ub = [1.0, 1.0])
@time res = solve(prob, Optimization.LBFGS(), maxiters = 100)
@test res.retcode == Optimization.SciMLBase.ReturnCode.Success

function con2_c(res, x, p)
    res .= [x[1]^2 + x[2]^2, (x[2] * sin(x[1]) + x[1]) - 5]
end

optf = OptimizationFunction(rosenbrock, AutoZygote(), cons = con2_c)
prob = OptimizationProblem(optf, x0, lcons = [1.0, -Inf],
    ucons = [1.0, 0.0], lb = [-1.0, -1.0],
    ub = [1.0, 1.0])
@time res = solve(prob, Optimization.LBFGS(), maxiters = 100)
@test res.retcode == SciMLBase.ReturnCode.Success

using MLUtils, OptimizationOptimisers

x0 = (-pi):0.001:pi
y0 = sin.(x0)
data = MLUtils.DataLoader((x0, y0), batchsize = 126)
function loss(coeffs, data)
    ypred = [evalpoly(data[1][i], coeffs) for i in eachindex(data[1])]
    return sum(abs2, ypred .- data[2])
end

function cons1(res, coeffs, p = nothing)
    res[1] = coeffs[1] * coeffs[5] - 1
    return nothing
end

optf = OptimizationFunction(loss, AutoSparseForwardDiff(), cons = cons1)
callback = (st, l) -> (@show l; return false)

initpars = rand(5)
l0 = optf(initpars, (x0, y0))
prob = OptimizationProblem(optf, initpars, (x0, y0), lcons = [-Inf], ucons = [0.5],
    lb = [-10.0, -10.0, -10.0, -10.0, -10.0], ub = [10.0, 10.0, 10.0, 10.0, 10.0])
opt1 = solve(prob, Optimization.LBFGS(), maxiters = 1000, callback = callback)
@test opt1.objective < l0

prob = OptimizationProblem(optf, initpars, data, lcons = [-Inf], ucons = [1],
    lb = [-10.0, -10.0, -10.0, -10.0, -10.0], ub = [10.0, 10.0, 10.0, 10.0, 10.0])
opt = solve(
    prob, Optimization.AugLag(; inner = Adam()), maxiters = 10000, callback = callback)
@test opt.objective < l0

optf1 = OptimizationFunction(loss, AutoSparseForwardDiff())
prob1 = OptimizationProblem(optf1, rand(5), data)
sol1 = solve(prob1, OptimizationOptimisers.Adam(), maxiters = 1000, callback = callback)
@test sol1.objective < l0

# Test constraint bounds validation (issue #959)
@testset "Constraint bounds validation" begin
    # Test case from issue #959 - missing lcons and ucons should give helpful error
    rosenbrock_constrained(u, p) = (p[1] - u[1])^2 + p[2] * (u[2] - u[1]^2)^2

    function cons_missing_bounds!(out, x, p)
        out[1] = sum(x)
    end

    optf_missing = OptimizationFunction(
        rosenbrock_constrained, AutoForwardDiff(), cons = cons_missing_bounds!)
    prob_missing = OptimizationProblem(optf_missing, [-1, 1.0], [1.0, 100.0])

    # Test LBFGS
    @test_throws ArgumentError solve(prob_missing, Optimization.LBFGS())

    # Verify the error message is helpful
    try
        solve(prob_missing, Optimization.LBFGS())
    catch e
        @test isa(e, ArgumentError)
        @test occursin("lcons", e.msg)
        @test occursin("ucons", e.msg)
        @test occursin("OptimizationProblem", e.msg)
        @test occursin("Example:", e.msg)
    end

    # Test AugLag
    @test_throws ArgumentError solve(prob_missing, Optimization.AugLag())

    # Verify the error message is helpful for AugLag too
    try
        solve(prob_missing, Optimization.AugLag())
    catch e
        @test isa(e, ArgumentError)
        @test occursin("lcons", e.msg)
        @test occursin("ucons", e.msg)
        @test occursin("OptimizationProblem", e.msg)
        @test occursin("Example:", e.msg)
    end

    # Test that it works when lcons and ucons are provided
    prob_with_bounds = OptimizationProblem(
        optf_missing, [-1, 1.0], [1.0, 100.0], lcons = [-Inf], ucons = [0.0])
    # This should not throw an error (though it may not converge)
    sol = solve(prob_with_bounds, Optimization.LBFGS(), maxiters = 10)
    @test !isnothing(sol)
end
