using OptimizationCMAEvolutionStrategy, Optimization
using Test

@testset "OptimizationCMAEvolutionStrategy.jl" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 100.0]
    l1 = rosenbrock(x0, _p)
    f = OptimizationFunction(rosenbrock)
    prob = OptimizationProblem(f, x0, _p, lb = [-1.0, -1.0], ub = [0.8, 0.8])
    sol = solve(prob, CMAEvolutionStrategyOpt())
    @test 10 * sol.objective < l1
end
