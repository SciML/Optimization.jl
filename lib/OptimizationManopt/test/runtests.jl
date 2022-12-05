using OptimizationManopt
using Optimization
using Manifolds
using ForwardDiff
using Manopt
using Test

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2

@testset "Gradient descent" begin
    x0 = zeros(2)
    p = [1.0, 100.0]

    stepsize = Manopt.ArmijoLinesearch(1.0,
                                       ExponentialRetraction(),
                                       0.5,
                                       0.0001)
    opt = OptimizationManopt.GradientDescentOptimizer(Euclidean(2),
                                                      stepsize = stepsize)

    optprob = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(optprob, x0, p)

    sol = Optimization.solve(prob, opt)
end

@testset "Nelder-Mead" begin
    x0 = zeros(2)
    p = [1.0, 100.0]

    opt = OptimizationManopt.NelderMeadOptimizer(Euclidean(2))

    optprob = OptimizationFunction(rosenbrock)
    prob = OptimizationProblem(optprob, x0, p)

    sol = Optimization.solve(prob, opt)
end
