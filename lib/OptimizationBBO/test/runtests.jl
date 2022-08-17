using OptimizationBBO, Optimization
using Test

@testset "OptimizationBBO.jl" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 100.0]
    l1 = rosenbrock(x0, _p)

    optprob = OptimizationFunction(rosenbrock)
    prob = Optimization.OptimizationProblem(optprob, x0, _p, lb = [-1.0, -1.0],
                                            ub = [0.8, 0.8])
    sol = solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited())
    @test 10 * sol.minimum < l1
end
