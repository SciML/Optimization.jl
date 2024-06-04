using OptimizationManopt
using Optimization
using Manifolds
using ForwardDiff, Zygote, Enzyme, FiniteDiff
using Manopt, RipQP, QuadraticModels
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
    @test sol.minimum < 0.7
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

@testset "CMA-ES" begin
    x0 = zeros(2)
    p = [1.0, 100.0]

    opt = OptimizationManopt.CMAESOptimizer()

    optprob = OptimizationFunction(rosenbrock)
    prob = OptimizationProblem(optprob, x0, p; manifold = R2)

    sol = Optimization.solve(prob, opt)
    @test sol.minimum < 0.1
end

@testset "ConvexBundle" begin
    x0 = zeros(2)
    p = [1.0, 100.0]

    opt = OptimizationManopt.ConvexBundleOptimizer()

    optprob = OptimizationFunction(rosenbrock, AutoForwardDiff())
    prob = OptimizationProblem(optprob, x0, p; manifold = R2)

    sol = Optimization.solve(
        prob, opt, sub_problem = Manopt.convex_bundle_method_subsolver!)
    @test sol.minimum < 0.1
end

# @testset "TruncatedConjugateGradientDescent" begin
#     x0 = zeros(2)
#     p = [1.0, 100.0]

#     opt = OptimizationManopt.TruncatedConjugateGradientDescentOptimizer()

#     optprob = OptimizationFunction(rosenbrock, AutoForwardDiff())
#     prob = OptimizationProblem(optprob, x0, p; manifold = R2)

#     sol = Optimization.solve(prob, opt)
#     @test_broken sol.minimum < 0.1
# end

@testset "AdaptiveRegularizationCubic" begin
    x0 = zeros(2)
    p = [1.0, 100.0]

    opt = OptimizationManopt.AdaptiveRegularizationCubicOptimizer()

    optprob = OptimizationFunction(rosenbrock, AutoForwardDiff())
    prob = OptimizationProblem(optprob, x0, p; manifold = R2)

    @test_broken Optimization.solve(prob, opt)
    @test_broken sol.minimum < 0.1
end

@testset "TrustRegions" begin
    x0 = zeros(2)
    p = [1.0, 100.0]

    opt = OptimizationManopt.TrustRegionsOptimizer()

    optprob = OptimizationFunction(rosenbrock, AutoForwardDiff())
    prob = OptimizationProblem(optprob, x0, p; manifold = R2)

    sol = Optimization.solve(prob, opt)
    @test sol.minimum < 0.1
end

# @testset "Circle example from Manopt" begin
#     Mc = Circle()
#     pc = 0.0
#     data = [-π / 4, 0.0, π / 4]
#     fc(y, _) = 1 / 2 * sum([distance(M, y, x)^2 for x in data])
#     sgrad_fc(G, y, _) = G .= -log(Mc, y, rand(data))

#     opt = OptimizationManopt.StochasticGradientDescentOptimizer()

#     optprob = OptimizationFunction(fc, grad = sgrad_fc)
#     prob = OptimizationProblem(optprob, pc; manifold = Mc)

#     sol = Optimization.solve(prob, opt)

#     @test all([is_point(Mc, q, true) for q in [q1, q2, q3, q4, q5]])
# end

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

    optf = OptimizationFunction(f, Optimization.AutoFiniteDiff())
    prob = OptimizationProblem(optf, data2[1]; manifold = M, maxiters = 1000)

    opt = OptimizationManopt.GradientDescentOptimizer()
    @time sol = Optimization.solve(prob, opt)

    @test sol.u≈q atol=1e-2

    function closed_form_solution!(M::SymmetricPositiveDefinite, q, L, U, p, X)
        # extract p^1/2 and p^{-1/2}
        (p_sqrt_inv, p_sqrt) = Manifolds.spd_sqrt_and_sqrt_inv(p)
        # Compute D & Q
        e2 = eigen(p_sqrt_inv * X * p_sqrt_inv) # decompose Sk  = QDQ'
        D = Diagonal(1.0 .* (e2.values .< 0))
        Q = e2.vectors
        Uprime = Q' * p_sqrt_inv * U * p_sqrt_inv * Q
        Lprime = Q' * p_sqrt_inv * L * p_sqrt_inv * Q
        P = cholesky(Hermitian(Uprime - Lprime))

        z = P.U' * D * P.U + Lprime
        copyto!(M, q, p_sqrt * Q * z * Q' * p_sqrt)
        return q
    end
    N = m
    U = mean(data2)
    L = inv(sum(1 / N * inv(matrix) for matrix in data2))

    opt = OptimizationManopt.FrankWolfeOptimizer()
    optf = OptimizationFunction(f, Optimization.AutoFiniteDiff())
    prob = OptimizationProblem(optf, data2[1]; manifold = M)

    @time sol = Optimization.solve(
        prob, opt, sub_problem = (M, q, p, X) -> closed_form_solution!(M, q, L, U, p, X),
        maxiters = 1000)
    @test sol.u≈q atol=1e-2
end
