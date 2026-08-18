using SimpleOptimization, OptimizationBase
using StaticArrays: SVector
using Test

@testset "SimpleOptimization.jl" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = [0.5, 0.5]
    x0s = SVector{2}(0.5, 0.5)
    p = [1.0, 100.0]
    l1 = rosenbrock(x0, p)

    @testset "ForwardDiff" begin
        using ForwardDiff
        optf = OptimizationFunction(rosenbrock, OptimizationBase.AutoForwardDiff())
        optfs = OptimizationFunction{false}(rosenbrock, OptimizationBase.AutoForwardDiff())
        prob = OptimizationProblem(optf, x0, p)
        probs = OptimizationProblem{false}(optfs, x0s, p)

        sol = solve(prob, SimpleLBFGS())
        @test sol.objective < l1
        @test sol.u ≈ [1.0, 1.0] atol = 1.0e-4
        @test sol.retcode == ReturnCode.Success
        @test typeof(sol.u) == typeof(x0)

        # Static path: SVector u0 stays SVector, no conversion.
        sols = solve(probs, SimpleLBFGS())
        @test sols.u ≈ [1.0, 1.0] atol = 1.0e-4
        @test typeof(sols.u) == typeof(x0s)

        sol_maxiters = solve(prob, SimpleLBFGS(); maxiters = 1)
        @test sol_maxiters.retcode == ReturnCode.MaxIters
        @test typeof(sol_maxiters.u) == typeof(x0)

        prob_box = OptimizationProblem(optf, x0, p; lb = [-2.0, -2.0], ub = [2.0, 2.0])
        sol = solve(prob_box, SimpleLBFGS())
        @test sol.u ≈ [1.0, 1.0] atol = 1.0e-4
        @test sol.retcode == ReturnCode.Success
        @test all(-2 .≤ sol.u) && all(sol.u .≤ 2)
        @test typeof(sol.u) == typeof(x0)

        prob_active = OptimizationProblem(optf, x0, p; lb = [-2.0, -2.0], ub = [0.8, 2.0])
        sol_active = solve(prob_active, SimpleLBFGS())
        @test sol_active.u[1] ≤ 0.8 + 1.0e-8
        @test sol_active.u[1] ≈ 0.8 atol = 1.0e-4
        @test sol_active.retcode == ReturnCode.Success
        @test typeof(sol_active.u) == typeof(x0)

        sol = solve(prob, SimpleBFGS())
        @test sol.objective < l1

        sol = solve(prob, SimpleGradientDescent(; eta = 0.001), maxiters = 10000)
        @test sol.objective < l1

        sol = solve(prob, SimpleNewton())
        @test sol.objective < l1

        sol = solve(prob, SimpleSOAP(; eta = 0.01), maxiters = 1000)
        @test sol.objective < l1

        @testset "SimpleSOAP Matrix" begin
            matrix_obj(X, P) = sum(abs2, X .- P)
            X0 = [1.0 2.0; 3.0 4.0]
            P_target = [0.0 0.0; 0.0 0.0]
            l1_mat = matrix_obj(X0, P_target)

            optf_mat = OptimizationFunction(matrix_obj, OptimizationBase.AutoForwardDiff())
            prob_mat = OptimizationProblem(optf_mat, X0, P_target)

            sol_mat = solve(prob_mat, SimpleSOAP(; eta = 0.1), maxiters = 500)

            @test sol_mat.objective < l1_mat
            @test sol_mat.objective < 1.0e-2
        end
    end
end
