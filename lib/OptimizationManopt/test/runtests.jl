using OptimizationManopt
using Optimization
using Manifolds
<<<<<<< HEAD
using ForwardDiff, Zygote, Enzyme
using Manopt
using Test
using Optimization.SciMLBase
using LinearAlgebra
=======
using ForwardDiff
using Manopt
using Test
using SciMLBase
>>>>>>> 84884237 (Move code from existing PR)

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2

function rosenbrock_grad!(storage, x, p)
    storage[1] = -2.0 * (p[1] - x[1]) - 4.0 * p[2] * (x[2] - x[1]^2) * x[1]
    storage[2] = 2.0 * p[2] * (x[2] - x[1]^2)
end

R2 = Euclidean(2)

<<<<<<< HEAD
@testset "Error on no or mismatching manifolds" begin
    x0 = zeros(2)
    p = [1.0, 100.0]

    stepsize = Manopt.ArmijoLinesearch(R2)
    opt = OptimizationManopt.GradientDescentOptimizer()

    optprob_forwarddiff = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
    prob_forwarddiff = OptimizationProblem(optprob_forwarddiff, x0, p)
    @test_throws ArgumentError("Manifold not specified in the problem for e.g. `OptimizationProblem(f, x, p; manifold = SymmetricPositiveDefinite(5))`.") Optimization.solve(
        prob_forwarddiff, opt)
end

=======
>>>>>>> 84884237 (Move code from existing PR)
@testset "Gradient descent" begin
    x0 = zeros(2)
    p = [1.0, 100.0]

    stepsize = Manopt.ArmijoLinesearch(R2)
<<<<<<< HEAD
    opt = OptimizationManopt.GradientDescentOptimizer()

    optprob_forwarddiff = OptimizationFunction(rosenbrock, Optimization.AutoEnzyme())
    prob_forwarddiff = OptimizationProblem(
        optprob_forwarddiff, x0, p; manifold = R2, stepsize = stepsize)
=======
    opt = OptimizationManopt.GradientDescentOptimizer(R2,
                                                      stepsize = stepsize)

    optprob_forwarddiff = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
    prob_forwarddiff = OptimizationProblem(optprob_forwarddiff, x0, p)
>>>>>>> 84884237 (Move code from existing PR)
    sol = Optimization.solve(prob_forwarddiff, opt)
    @test sol.minimum < 0.2

    optprob_grad = OptimizationFunction(rosenbrock; grad = rosenbrock_grad!)
<<<<<<< HEAD
    prob_grad = OptimizationProblem(optprob_grad, x0, p; manifold = R2, stepsize = stepsize)
=======
    prob_grad = OptimizationProblem(optprob_grad, x0, p)
>>>>>>> 84884237 (Move code from existing PR)
    sol = Optimization.solve(prob_grad, opt)
    @test sol.minimum < 0.2
end

@testset "Nelder-Mead" begin
    x0 = zeros(2)
    p = [1.0, 100.0]

<<<<<<< HEAD
    opt = OptimizationManopt.NelderMeadOptimizer()

    optprob = OptimizationFunction(rosenbrock)
    prob = OptimizationProblem(optprob, x0, p; manifold = R2)

    sol = Optimization.solve(prob, opt)
    @test sol.minimum < 1e-6
=======
    opt = OptimizationManopt.NelderMeadOptimizer(R2, [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

    optprob = OptimizationFunction(rosenbrock)
    prob = OptimizationProblem(optprob, x0, p)

    sol = Optimization.solve(prob, opt)
    @test sol.minimum < 0.7
>>>>>>> 84884237 (Move code from existing PR)
end

@testset "Conjugate gradient descent" begin
    x0 = zeros(2)
    p = [1.0, 100.0]

    stepsize = Manopt.ArmijoLinesearch(R2)
    opt = OptimizationManopt.ConjugateGradientDescentOptimizer(R2,
                                                               stepsize = stepsize)

    optprob = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(optprob, x0, p)

    sol = Optimization.solve(prob, opt)
    @test sol.minimum < 0.2
end

@testset "Quasi Newton" begin
    x0 = zeros(2)
    p = [1.0, 100.0]

    opt = OptimizationManopt.QuasiNewtonOptimizer(R2)

    optprob = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(optprob, x0, p)

    sol = Optimization.solve(prob, opt)
    @test sol.minimum < 1e-16
end

@testset "Particle swarm" begin
    x0 = zeros(2)
    p = [1.0, 100.0]

    opt = OptimizationManopt.ParticleSwarmOptimizer(R2)

    optprob = OptimizationFunction(rosenbrock)
    prob = OptimizationProblem(optprob, x0, p)

    sol = Optimization.solve(prob, opt)
    @test sol.minimum < 0.1
end

@testset "Custom constraints" begin
    cons(res, x, p) = (res .= [x[1]^2 + x[2]^2, x[1] * x[2]])

    x0 = zeros(2)
    p = [1.0, 100.0]
    opt = OptimizationManopt.GradientDescentOptimizer(R2)

    optprob_cons = OptimizationFunction(rosenbrock; grad = rosenbrock_grad!, cons = cons)
    prob_cons = OptimizationProblem(optprob_cons, x0, p)
    @test_throws SciMLBase.IncompatibleOptimizerError Optimization.solve(prob_cons, opt)
end
