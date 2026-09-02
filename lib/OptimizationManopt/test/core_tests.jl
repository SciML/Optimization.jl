using OptimizationManopt
using OptimizationBase
using Manifolds
using ForwardDiff, Zygote, Enzyme, FiniteDiff, ReverseDiff
using DifferentiationInterface: SecondOrder
using Manopt, ManifoldDiff, RipQP, QuadraticModels
using Test
using SciMLBase
using LinearAlgebra

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2

function rosenbrock_grad!(storage, x, p)
    storage[1] = -2.0 * (p[1] - x[1]) - 4.0 * p[2] * (x[2] - x[1]^2) * x[1]
    return storage[2] = 2.0 * p[2] * (x[2] - x[1]^2)
end

R2 = Euclidean(2)
@testset "OptimizationManopt.jl" begin
    @testset "Error on no or mismatching manifolds" begin
        x0 = zeros(2)
        p = [1.0, 100.0]

        stepsize = Manopt.ArmijoLinesearch(R2)
        opt = OptimizationManopt.GradientDescentOptimizer()

        optprob_forwarddiff = OptimizationFunction(rosenbrock, OptimizationBase.AutoForwardDiff())
        prob_forwarddiff = OptimizationProblem(optprob_forwarddiff, x0, p)
        @test_throws ArgumentError("Manifold not specified in the problem for e.g. `OptimizationProblem(f, x, p; manifold = SymmetricPositiveDefinite(5))`.") OptimizationBase.solve(
            prob_forwarddiff, opt
        )
    end

    @testset "Gradient descent" begin
        x0 = zeros(2)
        p = [1.0, 100.0]

        stepsize = Manopt.ArmijoLinesearch(R2)
        opt = OptimizationManopt.GradientDescentOptimizer()

        optprob_forwarddiff = OptimizationFunction(rosenbrock, OptimizationBase.AutoEnzyme())
        prob_forwarddiff = OptimizationProblem(
            optprob_forwarddiff, x0, p; manifold = R2, stepsize = stepsize
        )
        sol = OptimizationBase.solve(prob_forwarddiff, opt)
        @test sol.objective < 0.2

        optprob_grad = OptimizationFunction(rosenbrock; grad = rosenbrock_grad!)
        prob_grad = OptimizationProblem(optprob_grad, x0, p; manifold = R2, stepsize = stepsize)
        sol = OptimizationBase.solve(prob_grad, opt)
        @test sol.objective < 0.2
    end

    @testset "Nelder-Mead" begin
        x0 = zeros(2)
        p = [1.0, 100.0]

        opt = OptimizationManopt.NelderMeadOptimizer()

        optprob = OptimizationFunction(rosenbrock)
        prob = OptimizationProblem(optprob, x0, p; manifold = R2)

        sol = OptimizationBase.solve(prob, opt)
        @test sol.objective < 0.7
    end

    @testset "Conjugate gradient descent" begin
        x0 = zeros(2)
        p = [1.0, 100.0]

        stepsize = Manopt.ArmijoLinesearch(R2)
        opt = OptimizationManopt.ConjugateGradientDescentOptimizer()

        optprob = OptimizationFunction(rosenbrock, OptimizationBase.AutoForwardDiff())
        prob = OptimizationProblem(optprob, x0, p; manifold = R2)

        sol = OptimizationBase.solve(prob, opt, stepsize = stepsize)
        @test sol.objective < 0.5
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
        optprob = OptimizationFunction(rosenbrock, OptimizationBase.AutoForwardDiff())
        prob = OptimizationProblem(optprob, x0, p; manifold = R2)

        sol = OptimizationBase.solve(prob, opt, callback = callback, maxiters = 30)
        @test sol.objective < 1.0e-14
    end

    @testset "Particle swarm" begin
        x0 = zeros(2)
        p = [1.0, 100.0]

        opt = OptimizationManopt.ParticleSwarmOptimizer()

        optprob = OptimizationFunction(rosenbrock)
        prob = OptimizationProblem(optprob, x0, p; manifold = R2)

        sol = OptimizationBase.solve(prob, opt)
        @test sol.objective < 0.1
    end

    @testset "CMA-ES" begin
        x0 = zeros(2)
        p = [1.0, 100.0]

        opt = OptimizationManopt.CMAESOptimizer()

        optprob = OptimizationFunction(rosenbrock)
        prob = OptimizationProblem(optprob, x0, p; manifold = R2)

        sol = OptimizationBase.solve(prob, opt)
        @test sol.objective < 0.1
    end

    @testset "ConvexBundle" begin
        x0 = zeros(2)
        p = [1.0, 100.0]

        opt = OptimizationManopt.ConvexBundleOptimizer()

        optprob = OptimizationFunction(rosenbrock, AutoForwardDiff())
        prob = OptimizationProblem(optprob, x0, p; manifold = R2)

        sol = OptimizationBase.solve(
            prob, opt, sub_problem = Manopt.convex_bundle_method_subsolver
        )
        @test sol.objective < 0.1
    end

    # @testset "TruncatedConjugateGradientDescent" begin
    #     x0 = zeros(2)
    #     p = [1.0, 100.0]

    #     opt = OptimizationManopt.TruncatedConjugateGradientDescentOptimizer()

    #     optprob = OptimizationFunction(rosenbrock, AutoForwardDiff())
    #     prob = OptimizationProblem(optprob, x0, p; manifold = R2)

    #     sol = OptimizationBase.solve(prob, opt)
    #     @test_broken sol.objective < 0.1
    # end

    @testset "AdaptiveRegularizationCubic" begin
        x0 = zeros(2)
        p = [1.0, 100.0]

        opt = OptimizationManopt.AdaptiveRegularizationCubicOptimizer()

        optprob = OptimizationFunction(rosenbrock, SecondOrder(AutoForwardDiff(), AutoForwardDiff()))
        prob = OptimizationProblem(optprob, x0, p; manifold = R2)

        sol = OptimizationBase.solve(prob, opt)
        @test sol.objective < 0.1
        @test SciMLBase.successful_retcode(sol)
    end

    @testset "TrustRegions" begin
        x0 = zeros(2)
        p = [1.0, 100.0]

        opt = OptimizationManopt.TrustRegionsOptimizer()

        optprob = OptimizationFunction(rosenbrock, SecondOrder(AutoForwardDiff(), AutoForwardDiff()))
        prob = OptimizationProblem(optprob, x0, p; manifold = R2)

        sol = OptimizationBase.solve(prob, opt)
        @test sol.objective < 0.1
        @test SciMLBase.successful_retcode(sol)
    end

    # Second-order solvers on a manifold with matrix-valued points. The Euclidean Hessian
    # buffers used to be allocated as flat `length(θ)` vectors, which broke the
    # `riemannian_Hessian!` projection for anything but vector-shaped points (#1036).
    @testset "Hessian on matrix manifolds" begin
        A = [
            4.0 1.0 0.0 0.0 0.0
            1.0 3.0 1.0 0.0 0.0
            0.0 1.0 2.0 1.0 0.0
            0.0 0.0 1.0 1.0 1.0
            0.0 0.0 0.0 1.0 5.0
        ]
        St = Stiefel(5, 2)
        # sum of the two smallest eigenvalues of A is the minimum of tr(X'AX) on St
        target = sum(eigvals(Symmetric(A))[1:2])
        brockett(X, p) = tr(X' * p * X)
        egrad!(G, X, p) = (G .= 2 * p * X)
        ehess!(H, X, p) = (H .= kron(I(2), 2 * p))
        X0 = Matrix(qr([1.0 0.0; 0.0 1.0; 1.0 1.0; 0.0 1.0; 1.0 0.0]).Q)[:, 1:2]

        for opt in (
                OptimizationManopt.TrustRegionsOptimizer(),
                OptimizationManopt.AdaptiveRegularizationCubicOptimizer(),
            )
            @testset "$(nameof(typeof(opt)))" begin
                # Euclidean Hessian-vector product from AD
                optf = OptimizationFunction(
                    brockett, SecondOrder(AutoForwardDiff(), AutoForwardDiff())
                )
                sol = OptimizationBase.solve(
                    OptimizationProblem(optf, X0, A; manifold = St), opt
                )
                @test SciMLBase.successful_retcode(sol)
                @test sol.objective ≈ target atol = 1.0e-6
                @test is_point(St, sol.u; error = :none)

                # User-supplied dense Euclidean Hessian, no `hv`
                optf = OptimizationFunction(brockett; grad = egrad!, hess = ehess!)
                sol = OptimizationBase.solve(
                    OptimizationProblem(optf, X0, A; manifold = St), opt
                )
                @test SciMLBase.successful_retcode(sol)
                @test sol.objective ≈ target atol = 1.0e-6

                # Gradient only: Manopt falls back to its approximate Hessian
                optf = OptimizationFunction(brockett; grad = egrad!)
                sol = OptimizationBase.solve(
                    OptimizationProblem(optf, X0, A; manifold = St), opt
                )
                @test SciMLBase.successful_retcode(sol)
                @test sol.objective ≈ target atol = 1.0e-6
            end
        end

        # The converted Hessian agrees with `riemannian_Hessian` fed the exact Euclidean
        # gradient and Hessian-vector product, and is a tangent vector.
        optf = OptimizationFunction(brockett, SecondOrder(AutoForwardDiff(), AutoForwardDiff()))
        cache = OptimizationBase.init(
            OptimizationProblem(optf, X0, A; manifold = St),
            OptimizationManopt.TrustRegionsOptimizer()
        )
        hessF = OptimizationManopt.build_hessF(cache.f)
        Xt = project(St, X0, [1.0 0.0; 0.0 -1.0; 0.5 0.5; 0.0 0.0; -1.0 1.0])
        Y = hessF(St, X0, Xt)
        Yref = ManifoldDiff.riemannian_Hessian(St, X0, 2 * A * X0, 2 * A * Xt, Xt)
        @test Y ≈ Yref
        @test is_vector(St, X0, Y; error = :none)
        Y2 = zero(X0)
        hessF(St, Y2, X0, Xt)
        @test Y2 ≈ Yref

        # No second-order information at all yields `nothing`, not a closure that throws
        optf_g = OptimizationFunction(brockett; grad = egrad!)
        cache_g = OptimizationBase.init(
            OptimizationProblem(optf_g, X0, A; manifold = St),
            OptimizationManopt.TrustRegionsOptimizer()
        )
        @test OptimizationManopt.build_hessF(cache_g.f) === nothing
    end

    @testset "Custom constraints" begin
        cons(res, x, p) = (res .= [x[1]^2 + x[2]^2, x[1] * x[2]])

        x0 = zeros(2)
        p = [1.0, 100.0]
        opt = OptimizationManopt.GradientDescentOptimizer()

        optprob_cons = OptimizationFunction(rosenbrock; grad = rosenbrock_grad!, cons = cons)
        prob_cons = OptimizationProblem(optprob_cons, x0, p)
        #TODO: What is this?
        @test_throws OptimizationBase.IncompatibleOptimizerError OptimizationBase.solve(prob_cons, opt)
    end
end
