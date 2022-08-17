using OptimizationEvolutionary, Optimization
using Test

@testset "OptimizationEvolutionary.jl" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 100.0]
    l1 = rosenbrock(x0, _p)
    optprob = OptimizationFunction(rosenbrock)
    prob = Optimization.OptimizationProblem(optprob, x0, _p)
    sol = solve(prob, CMAES(μ = 40, λ = 100), abstol = 1e-15)
    @test 10 * sol.minimum < l1
end
