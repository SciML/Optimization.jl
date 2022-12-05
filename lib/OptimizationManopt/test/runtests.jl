using OptimizationManopt
using Optimization
using Manifolds
using ForwardDiff
using Manopt
using Test

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2

R2 = Euclidean(2)

@testset "Gradient descent" begin
    x0 = zeros(2)
    p = [1.0, 100.0]

    stepsize = Manopt.ArmijoLinesearch(R2)
    opt = OptimizationManopt.GradientDescentOptimizer(R2,
                                                      stepsize = stepsize)

    optprob = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(optprob, x0, p)

    sol = Optimization.solve(prob, opt)
    @test sol.minimum < 0.2
end

@testset "Nelder-Mead" begin
    x0 = zeros(2)
    p = [1.0, 100.0]

    opt = OptimizationManopt.NelderMeadOptimizer(R2, [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

    optprob = OptimizationFunction(rosenbrock)
    prob = OptimizationProblem(optprob, x0, p)

    sol = Optimization.solve(prob, opt)
    @test sol.minimum < 0.7
end
