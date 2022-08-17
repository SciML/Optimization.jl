using OptimizationPolyalgorithms, Optimization, ForwardDiff
using Test

@testset "OptimizationPolyalgorithms.jl" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 100.0]
    l1 = rosenbrock(x0, _p)

    optprob = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(optprob, x0, _p)
    sol = Optimization.solve(prob, PolyOpt(), maxiters = 1000)
    @test 10 * sol.minimum < l1
end
