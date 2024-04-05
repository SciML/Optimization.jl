using OptimizationManopt
using Optimization
using Manifolds
using ForwardDiff, Zygote, Enzyme
using Manopt
using Test
using Optimization.SciMLBase
using LinearAlgebra

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2

function rosenbrock_grad!(storage, x, p)
    storage[1] = -2.0 * (p[1] - x[1]) - 4.0 * p[2] * (x[2] - x[1]^2) * x[1]
    storage[2] = 2.0 * p[2] * (x[2] - x[1]^2)
end

R2 = Euclidean(2)

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

@testset "Gradient descent" begin
    x0 = zeros(2)
    p = [1.0, 100.0]

    stepsize = Manopt.ArmijoLinesearch(R2)
    opt = OptimizationManopt.GradientDescentOptimizer()

    optprob_forwarddiff = OptimizationFunction(rosenbrock, Optimization.AutoEnzyme())
    prob_forwarddiff = OptimizationProblem(
        optprob_forwarddiff, x0, p; manifold = R2, stepsize = stepsize)
    sol = Optimization.solve(prob_forwarddiff, opt)
    @test sol.minimum < 0.2

    optprob_grad = OptimizationFunction(rosenbrock; grad = rosenbrock_grad!)
    prob_grad = OptimizationProblem(optprob_grad, x0, p; manifold = R2, stepsize = stepsize)
    sol = Optimization.solve(prob_grad, opt)
    @test sol.minimum < 0.2
end

@testset "Nelder-Mead" begin
    x0 = zeros(2)
    p = [1.0, 100.0]

    opt = OptimizationManopt.NelderMeadOptimizer()

    optprob = OptimizationFunction(rosenbrock)
    prob = OptimizationProblem(optprob, x0, p; manifold = R2)

    sol = Optimization.solve(prob, opt)
    @test sol.minimum < 1e-6
end

@testset "Conjugate gradient descent" begin
    x0 = zeros(2)
    p = [1.0, 100.0]

    stepsize = Manopt.ArmijoLinesearch(R2)
    opt = OptimizationManopt.ConjugateGradientDescentOptimizer()

    optprob = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(optprob, x0, p; manifold = R2)

    sol = Optimization.solve(prob, opt, stepsize = stepsize)
    @test sol.minimum < 0.5
end

@testset "Quasi Newton" begin
    x0 = zeros(2)
    p = [1.0, 100.0]

    opt = OptimizationManopt.QuasiNewtonOptimizer()
    function callback(state, l)
        println(state.u)
        println(l)
        return false
    end
    optprob = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(optprob, x0, p; manifold = R2)

    sol = Optimization.solve(prob, opt, callback = callback, maxiters = 30)
    @test sol.minimum < 1e-14
end

@testset "Particle swarm" begin
    x0 = zeros(2)
    p = [1.0, 100.0]

    opt = OptimizationManopt.ParticleSwarmOptimizer()

    optprob = OptimizationFunction(rosenbrock)
    prob = OptimizationProblem(optprob, x0, p; manifold = R2)

    sol = Optimization.solve(prob, opt)
    @test sol.minimum < 0.1
end

@testset "Custom constraints" begin
    cons(res, x, p) = (res .= [x[1]^2 + x[2]^2, x[1] * x[2]])

    x0 = zeros(2)
    p = [1.0, 100.0]
    opt = OptimizationManopt.GradientDescentOptimizer()

    optprob_cons = OptimizationFunction(rosenbrock; grad = rosenbrock_grad!, cons = cons)
    prob_cons = OptimizationProblem(optprob_cons, x0, p)
    @test_throws SciMLBase.IncompatibleOptimizerError Optimization.solve(prob_cons, opt)
end

@testset "SPD Manifold" begin
    M = SymmetricPositiveDefinite(5)
    m = 100
    σ = 0.005
    q = Matrix{Float64}(I, 5, 5) .+ 2.0
    data2 = [exp(M, q, σ * rand(M; vector_at = q)) for i in 1:m]

    f(M, x, p = nothing) = sum(distance(M, x, data2[i])^2 for i in 1:m)
    f(x, p = nothing) = sum(distance(M, x, data2[i])^2 for i in 1:m)

    optf = OptimizationFunction(f, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(optf, data2[1]; manifold = M, maxiters = 1000)

    opt = OptimizationManopt.GradientDescentOptimizer()
    @time sol = Optimization.solve(prob, opt)

    @test sol.u≈q atol=1e-2
end
