using OptimizationManopt
using OptimizationBase
using Manifolds
using ForwardDiff, Zygote, Enzyme, FiniteDiff, ReverseDiff
using DifferentiationInterface: SecondOrder
using Manopt, RipQP, QuadraticModels
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

    @testset "Gradient descent maxiters without abstol still reports Success" begin
        # Regression test: passing only `maxiters` (no `abstol`) used to fully replace
        # Manopt's own default `stopping_criterion`, leaving `StopAfterIteration` as the
        # *only* criterion. Since that criterion never `indicates_convergence`, the run
        # was structurally guaranteed to report `ReturnCode.MaxIters`, even once it had
        # actually converged. `maxiters` here is set well above what's needed so the run
        # converges before hitting the cap.
        x0 = zeros(2)
        p = [1.0, 100.0]

        stepsize = Manopt.ArmijoLinesearch(R2)
        opt = OptimizationManopt.GradientDescentOptimizer()

        optprob = OptimizationFunction(rosenbrock, OptimizationBase.AutoForwardDiff())
        prob = OptimizationProblem(
            optprob, x0, p; manifold = R2, stepsize = stepsize, maxiters = 25_000
        )
        sol = OptimizationBase.solve(prob, opt)
        @test sol.objective < 1.0e-10
        @test SciMLBase.successful_retcode(sol)
    end

    @testset "User-supplied stopping_criterion is combined, not clobbered" begin
        # Regression test: a Manopt `stopping_criterion` passed through `solve` used to be
        # silently overwritten whenever `maxiters`/`maxtime`/`abstol` was also given.
        x0 = zeros(2)
        p = [1.0, 100.0]

        stepsize = Manopt.ArmijoLinesearch(R2)
        opt = OptimizationManopt.GradientDescentOptimizer()

        optprob = OptimizationFunction(rosenbrock, OptimizationBase.AutoForwardDiff())
        prob = OptimizationProblem(
            optprob, x0, p; manifold = R2, stepsize = stepsize, maxiters = 25_000
        )
        user_sc = Manopt.StopWhenGradientNormLess(1.0e-3)
        sol = OptimizationBase.solve(prob, opt; stopping_criterion = user_sc)
        sc = Manopt.get_stopping_criterion(sol.original)
        @test any(c -> c === user_sc, sc.criteria)
        # The user's criterion also suppresses the (tighter) 1e-8 default fallback.
        @test !any(c -> c isa Manopt.StopWhenGradientNormLess && c !== user_sc, sc.criteria)
        @test SciMLBase.successful_retcode(sol)
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
        # With `maxiters` given, Manopt's default `StopWhenGradientNormLess(1e-6)` stays
        # active, so the run stops (converged) slightly earlier than the old
        # `StopAfterIteration`-only criterion, which ran the budget down to < 1e-14.
        @test sol.objective < 1.0e-12
        @test SciMLBase.successful_retcode(sol)
    end

    @testset "Particle swarm" begin
        x0 = zeros(2)
        p = [1.0, 100.0]

        opt = OptimizationManopt.ParticleSwarmOptimizer()

        optprob = OptimizationFunction(rosenbrock)
        prob = OptimizationProblem(optprob, x0, p; manifold = R2)

        sol = OptimizationBase.solve(prob, opt)
        @test sol.objective < 0.1

        # Manopt's own PSO convergence test (`StopWhenSwarmVelocityLess`) never
        # `indicates_convergence`, so no fallback criterion is injected for PSO and a
        # `maxiters`-bounded run keeps reporting `MaxIters` instead of stopping early on
        # some unrelated criterion and reporting `Failure`/spurious `Success`.
        sol = OptimizationBase.solve(prob, opt; maxiters = 100)
        @test sol.retcode == SciMLBase.ReturnCode.MaxIters
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

        #TODO: This autodiff currently provides a Hessian that seem to not provide a Hessian
        # ARC Fails but also AD before that warns. So it passes _some_ hessian but a wrong one, even in format
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

        #TODO: This autodiff currently provides a Hessian that seem to not provide a Hessian
        # TR Fails but also AD before that warns. So it passes _some_ hessian but a wrong one, even in format
        optprob = OptimizationFunction(rosenbrock, SecondOrder(AutoForwardDiff(), AutoForwardDiff()))
        prob = OptimizationProblem(optprob, x0, p; manifold = R2)

        sol = OptimizationBase.solve(prob, opt)
        @test sol.objective < 0.1
        @test SciMLBase.successful_retcode(sol)
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
