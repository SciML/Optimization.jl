using OptimizationBase, OptimizationIpopt
using Zygote
using Test
using LinearAlgebra
using SparseArrays

# These tests were automatically translated from the Ipopt tests, https://github.com/coin-or/Ipopt
# licensed under Eclipse Public License - v 2.0
# https://github.com/coin-or/Ipopt/blob/stable/3.14/LICENSE

@testset "Advanced Ipopt Features" begin

    @testset "Custom Tolerances and Options" begin
        # Test setting various Ipopt-specific options
        rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2

        x0 = [0.0, 0.0]
        p = [1.0, 100.0]

        optfunc = OptimizationFunction(rosenbrock, OptimizationBase.AutoZygote())
        prob = OptimizationProblem(optfunc, x0, p)

        # Test with tight tolerances
        sol = solve(
            prob, IpoptOptimizer(
                acceptable_tol = 1.0e-8,
                acceptable_iter = 5
            );
            reltol = 1.0e-10
        )

        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ [1.0, 1.0] atol = 1.0e-8
    end

    @testset "Constraint Violation Tolerance" begin
        # Test problem with different constraint tolerances
        function obj(x, p)
            return x[1]^2 + x[2]^2
        end

        function cons(res, x, p)
            res[1] = x[1] + x[2] - 2.0
            res[2] = x[1]^2 + x[2]^2 - 2.0
        end

        optfunc = OptimizationFunction(obj, OptimizationBase.AutoZygote(); cons = cons)
        prob = OptimizationProblem(
            optfunc, [0.5, 0.5], nothing;
            lcons = [0.0, 0.0],
            ucons = [0.0, 0.0]
        )

        sol = solve(
            prob, IpoptOptimizer(
                constr_viol_tol = 1.0e-8
            )
        )

        @test SciMLBase.successful_retcode(sol)
        @test sol.u[1] + sol.u[2] ≈ 2.0 atol = 1.0e-7
        @test sol.u[1]^2 + sol.u[2]^2 ≈ 2.0 atol = 1.0e-7
    end

    @testset "Derivative Test" begin
        # Test with derivative checking enabled
        function complex_obj(x, p)
            return sin(x[1]) * cos(x[2]) + exp(-x[1]^2 - x[2]^2)
        end

        optfunc = OptimizationFunction(complex_obj, OptimizationBase.AutoZygote())
        prob = OptimizationProblem(optfunc, [0.1, 0.1], nothing)

        # Run with derivative test level 1 (first derivatives only)
        sol = solve(
            prob, IpoptOptimizer(
                additional_options = Dict(
                    "derivative_test" => "first-order",
                    "derivative_test_tol" => 1.0e-4
                )
            )
        )

        @test SciMLBase.successful_retcode(sol)
    end

    @testset "Linear Solver Options" begin
        # Test different linear solver options if available
        rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2

        x0 = zeros(10)  # Larger problem
        p = [1.0, 100.0]

        # Extend Rosenbrock to n dimensions
        function rosenbrock_n(x, p)
            n = length(x)
            sum = 0.0
            for i in 1:2:(n - 1)
                sum += (p[1] - x[i])^2 + p[2] * (x[i + 1] - x[i]^2)^2
            end
            return sum
        end

        optfunc = OptimizationFunction(rosenbrock_n, OptimizationBase.AutoZygote())
        prob = OptimizationProblem(optfunc, x0, p)

        # Test with different linear solver strategies
        sol = solve(
            prob, IpoptOptimizer(
                linear_solver = "mumps"
            )
        )  # or "ma27", "ma57", etc. if available

        @test SciMLBase.successful_retcode(sol)
        # Check that odd indices are close to 1
        @test all(isapprox(sol.u[i], 1.0, atol = 1.0e-4) for i in 1:2:(length(x0) - 1))
    end

    @testset "Scaling Options" begin
        # Test problem that benefits from scaling
        function scaled_obj(x, p)
            return 1.0e6 * x[1]^2 + 1.0e-6 * x[2]^2
        end

        function scaled_cons(res, x, p)
            res[1] = 1.0e3 * x[1] + 1.0e-3 * x[2] - 1.0
        end

        optfunc = OptimizationFunction(
            scaled_obj, OptimizationBase.AutoZygote();
            cons = scaled_cons
        )
        prob = OptimizationProblem(
            optfunc, [1.0, 1.0], nothing;
            lcons = [0.0],
            ucons = [0.0]
        )

        # Solve with automatic scaling
        sol = solve(
            prob, IpoptOptimizer(
                nlp_scaling_method = "gradient-based"
            )
        )

        @test SciMLBase.successful_retcode(sol)
        # Check constraint satisfaction
        res = zeros(1)
        scaled_cons(res, sol.u, nothing)
        @test abs(res[1]) < 1.0e-6
    end

    @testset "Restoration Phase Test" begin
        # Problem that might trigger restoration phase
        function difficult_obj(x, p)
            return x[1]^4 + x[2]^4
        end

        function difficult_cons(res, x, p)
            res[1] = x[1]^3 + x[2]^3 - 1.0
            res[2] = x[1]^2 + x[2]^2 - 0.5
        end

        optfunc = OptimizationFunction(
            difficult_obj, OptimizationBase.AutoZygote();
            cons = difficult_cons
        )
        # Start from an infeasible point
        prob = OptimizationProblem(
            optfunc, [2.0, 2.0], nothing;
            lcons = [0.0, 0.0],
            ucons = [0.0, 0.0]
        )

        sol = solve(
            prob, IpoptOptimizer(
                additional_options = Dict(
                    "required_infeasibility_reduction" => 0.9
                )
            )
        )

        if SciMLBase.successful_retcode(sol)
            # Check constraint satisfaction if successful
            res = zeros(2)
            difficult_cons(res, sol.u, nothing)
            @test norm(res) < 1.0e-4
        end
    end

    @testset "Mu Strategy Options" begin
        # Test different barrier parameter update strategies
        rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2

        x0 = [0.0, 0.0]
        p = [1.0, 100.0]

        optfunc = OptimizationFunction(rosenbrock, OptimizationBase.AutoZygote())
        prob = OptimizationProblem(optfunc, x0, p)

        # Test adaptive mu strategy
        sol = solve(
            prob, IpoptOptimizer(
                mu_strategy = "adaptive",
                mu_init = 0.1
            )
        )

        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ [1.0, 1.0] atol = 1.0e-4

        # Test monotone mu strategy
        sol2 = solve(
            prob, IpoptOptimizer(
                mu_strategy = "monotone"
            )
        )

        @test SciMLBase.successful_retcode(sol2)
        @test sol2.u ≈ [1.0, 1.0] atol = 1.0e-4
    end

    @testset "Fixed Variable Handling" begin
        # Test problem with effectively fixed variables
        function fixed_var_obj(x, p)
            return (x[1] - 1)^2 + (x[2] - 2)^2 + (x[3] - 3)^2
        end

        optfunc = OptimizationFunction(fixed_var_obj, OptimizationBase.AutoZygote())
        # Fix x[2] = 2.0 by setting equal bounds
        prob = OptimizationProblem(
            optfunc, [0.0, 2.0, 0.0], nothing;
            lb = [-Inf, 2.0, -Inf],
            ub = [Inf, 2.0, Inf]
        )

        sol = solve(
            prob, IpoptOptimizer(
                additional_options = Dict(
                    "fixed_variable_treatment" => "make_parameter"
                )
            )
        )

        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ [1.0, 2.0, 3.0] atol = 1.0e-6
    end

    @testset "Acceptable Point Termination" begin
        # Test reaching an acceptable point rather than optimal
        function slow_converge_obj(x, p)
            return sum(exp(-10 * (x[i] - i / 10)^2) for i in 1:length(x))
        end

        n = 5
        optfunc = OptimizationFunction(
            slow_converge_obj,
            OptimizationBase.AutoZygote()
        )
        prob = OptimizationProblem(
            optfunc, zeros(n), nothing;
            sense = OptimizationBase.MaxSense
        )

        sol = solve(
            prob, IpoptOptimizer(
                acceptable_tol = 1.0e-4,
                acceptable_iter = 10
            );
            maxiters = 50
        )

        @test SciMLBase.successful_retcode(sol)
    end
end

@testset "Output and Logging Options" begin
    # Test various output options
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2

    x0 = [0.0, 0.0]
    p = [1.0, 100.0]

    optfunc = OptimizationFunction(rosenbrock, OptimizationBase.AutoZygote())
    prob = OptimizationProblem(optfunc, x0, p)

    @testset "Verbose levels" begin
        for verbose_level in [false, 0, 3, 5]
            sol = solve(prob, IpoptOptimizer(); verbose = verbose_level)
            @test SciMLBase.successful_retcode(sol)
        end
    end

    @testset "Timing statistics" begin
        sol = solve(prob, IpoptOptimizer(print_timing_statistics = "yes"))
        @test SciMLBase.successful_retcode(sol)
    end
end
