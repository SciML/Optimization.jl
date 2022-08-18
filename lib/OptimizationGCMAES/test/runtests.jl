using OptimizationGCMAES, Optimization, ForwardDiff
using Test

@testset "OptimizationGCMAES.jl" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 100.0]
    l1 = rosenbrock(x0, _p)
    f_ad = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
    f_noad = OptimizationFunction(rosenbrock)

    prob = Optimization.OptimizationProblem(f_ad, x0, _p, lb = [-1.0, -1.0],
                                            ub = [1.0, 1.0])
    sol = solve(prob, GCMAESOpt(), maxiters = 1000)
    @test 10 * sol.minimum < l1

    prob = Optimization.OptimizationProblem(f_noad, x0, _p, lb = [-1.0, -1.0],
                                            ub = [1.0, 1.0])
    sol = solve(prob, GCMAESOpt(), maxiters = 1000)
    @test 10 * sol.minimum < l1
end
