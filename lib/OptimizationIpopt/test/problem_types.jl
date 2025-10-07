using OptimizationBase, OptimizationIpopt
using Zygote
using Test
using LinearAlgebra
using SparseArrays

# These tests were automatically translated from the Ipopt tests, https://github.com/coin-or/Ipopt
# licensed under Eclipse Public License - v 2.0
# https://github.com/coin-or/Ipopt/blob/stable/3.14/LICENSE

@testset "Specific Problem Types" begin

    @testset "Optimal Control Problem" begin
        # Discretized optimal control problem
        # minimize integral of u^2 subject to dynamics

        N = 20  # number of time steps
        dt = 0.1

        function control_objective(z, p)
            # z = [x1, x2, ..., xN, u1, u2, ..., uN-1]
            # Minimize control effort
            u_start = N + 1
            return sum(z[i]^2 for i in u_start:length(z))
        end

        function dynamics_constraints(res, z, p)
            # Enforce dynamics x[i+1] = x[i] + dt * u[i]
            for i in 1:N-1
                res[i] = z[i+1] - z[i] - dt * z[N + i]
            end
            # Initial condition
            res[N] = z[1] - 0.0
            # Final condition
            res[N+1] = z[N] - 1.0
        end

        n_vars = N + (N-1)  # states + controls
        n_cons = N + 1      # dynamics + boundary conditions

        optfunc = OptimizationFunction(control_objective, AutoZygote();
                                      cons = dynamics_constraints)
        z0 = zeros(n_vars)
        prob = OptimizationProblem(optfunc, z0;
                                 lcons = zeros(n_cons),
                                 ucons = zeros(n_cons))

        sol = solve(prob, IpoptOptimizer())

        @test SciMLBase.successful_retcode(sol)
        @test sol.u[1] ≈ 0.0 atol=1e-6  # Initial state
        @test sol.u[N] ≈ 1.0 atol=1e-6  # Final state
    end

    @testset "Portfolio Optimization" begin
        # Markowitz portfolio optimization with constraints
        # minimize risk (variance) subject to return constraint

        n_assets = 5
        # Expected returns (random for example)
        μ = [0.05, 0.10, 0.15, 0.08, 0.12]
        # Covariance matrix (positive definite)
        Σ = [0.05 0.01 0.02 0.01 0.00;
             0.01 0.10 0.03 0.02 0.01;
             0.02 0.03 0.15 0.02 0.03;
             0.01 0.02 0.02 0.08 0.02;
             0.00 0.01 0.03 0.02 0.06]

        target_return = 0.10

        function portfolio_risk(w, p)
            return dot(w, Σ * w)
        end

        function portfolio_constraints(res, w, p)
            # Sum of weights = 1
            res[1] = sum(w) - 1.0
            # Expected return >= target
            res[2] = dot(μ, w) - target_return
        end

        optfunc = OptimizationFunction(portfolio_risk, AutoZygote();
                                      cons = portfolio_constraints)
        w0 = fill(1.0/n_assets, n_assets)
        prob = OptimizationProblem(optfunc, w0;
                                 lb = zeros(n_assets),  # No short selling
                                 ub = ones(n_assets),   # No leverage
                                 lcons = [0.0, 0.0],
                                 ucons = [0.0, Inf])

        sol = solve(prob, IpoptOptimizer())

        @test SciMLBase.successful_retcode(sol)
        @test sum(sol.u) ≈ 1.0 atol=1e-6
        @test dot(μ, sol.u) >= target_return - 1e-6
        @test all(sol.u .>= -1e-6)  # Non-negative weights
    end

    @testset "Geometric Programming" begin
        # Geometric program in standard form
        # minimize c^T * x subject to geometric constraints

        function geometric_obj(x, p)
            # Objective: x1 * x2 * x3 (in log form: log(x1) + log(x2) + log(x3))
            return exp(x[1]) * exp(x[2]) * exp(x[3])
        end

        function geometric_cons(res, x, p)
            # Constraint: x1^2 * x2 / x3 <= 1
            # In exponential form: 2*x1 + x2 - x3 <= 0
            res[1] = exp(2*x[1] + x[2] - x[3]) - 1.0
            # Constraint: x1 + x2 + x3 = 1 (in exponential variables)
            res[2] = exp(x[1]) + exp(x[2]) + exp(x[3]) - 3.0
        end

        optfunc = OptimizationFunction(geometric_obj, AutoZygote();
                                      cons = geometric_cons)
        x0 = zeros(3)  # log variables start at 0 (original variables = 1)
        prob = OptimizationProblem(optfunc, x0;
                                 lcons = [-Inf, 0.0],
                                 ucons = [0.0, 0.0])

        sol = solve(prob, IpoptOptimizer())

        @test SciMLBase.successful_retcode(sol)
        # Check constraints
        res = zeros(2)
        geometric_cons(res, sol.u, nothing)
        @test res[1] <= 1e-6
        @test abs(res[2]) <= 1e-6
    end

    @testset "Parameter Estimation" begin
        # Nonlinear least squares parameter estimation
        # Fit exponential decay model: y = a * exp(-b * t) + c

        # Generate synthetic data
        true_params = [2.0, 0.5, 0.1]
        t_data = collect(0:0.5:5)
        y_data = @. true_params[1] * exp(-true_params[2] * t_data) + true_params[3]
        # Add noise
        # y_data += 0.03 * randn(length(t_data))
        y_data += [0.05, 0.01, 0.01, 0.025, 0.0001, 0.004, 0.0056, 0.003, 0.0076, 0.012, 0.0023]

        function residual_sum_squares(params, p)
            a, b, c = params
            residuals = @. y_data - (a * exp(-b * t_data) + c)
            return sum(residuals.^2)
        end

        optfunc = OptimizationFunction(residual_sum_squares, AutoZygote())
        # Initial guess
        params0 = [1.0, 1.0, 0.0]
        prob = OptimizationProblem(optfunc, params0;
                                 lb = [0.0, 0.0, -1.0],  # a, b > 0
                                 ub = [10.0, 10.0, 1.0])

        sol = solve(prob, IpoptOptimizer(
                   acceptable_tol = 1e-10);
                   reltol = 1e-10)

        @test SciMLBase.successful_retcode(sol)
        # Parameters should be close to true values (within noise)
        @test sol.u[1] ≈ true_params[1] atol=0.2
        @test sol.u[2] ≈ true_params[2] atol=0.1
        @test sol.u[3] ≈ true_params[3] atol=0.05
    end

    @testset "Network Flow Problem" begin
        # Minimum cost flow problem
        # Simple network: source -> 2 intermediate nodes -> sink

        # Network structure (4 nodes, 5 edges)
        # Node 1: source, Node 4: sink
        # Edges: (1,2), (1,3), (2,3), (2,4), (3,4)

        # Edge costs
        costs = [2.0, 3.0, 1.0, 4.0, 2.0]
        # Edge capacities
        capacities = [10.0, 8.0, 5.0, 10.0, 10.0]
        # Required flow from source to sink
        required_flow = 15.0

        function flow_cost(flows, p)
            return dot(costs, flows)
        end

        function flow_constraints(res, flows, p)
            # Conservation at node 2: flow in = flow out
            res[1] = flows[1] - flows[3] - flows[4]
            # Conservation at node 3: flow in = flow out
            res[2] = flows[2] + flows[3] - flows[5]
            # Total flow from source
            res[3] = flows[1] + flows[2] - required_flow
            # Total flow to sink
            res[4] = flows[4] + flows[5] - required_flow
        end

        optfunc = OptimizationFunction(flow_cost, OptimizationBase.AutoZygote();
                                      cons = flow_constraints)
        flows0 = fill(required_flow / 2, 5)
        prob = OptimizationProblem(optfunc, flows0, nothing;
                                 lb = zeros(5),
                                 ub = capacities,
                                 lcons = zeros(4),
                                 ucons = zeros(4))

        sol = solve(prob, IpoptOptimizer())

        @test SciMLBase.successful_retcode(sol)
        @test all(sol.u .>= -1e-6)  # Non-negative flows
        @test all(sol.u .<= capacities .+ 1e-6)  # Capacity constraints
        # Check flow conservation
        res = zeros(4)
        flow_constraints(res, sol.u, nothing)
        @test norm(res) < 1e-6
    end

    @testset "Robust Optimization" begin
        # Simple robust optimization problem
        # minimize worst-case objective over uncertainty set

        function robust_objective(x, p)
            # Minimize max_{u in U} (x - u)^T * (x - u)
            # where U = {u : ||u||_inf <= 0.5}
            # This simplifies to minimizing ||x||^2 + ||x||_1
            return sum(x.^2) + sum(abs.(x))
        end

        function robust_constraints(res, x, p)
            # Constraint: sum(x) >= 1
            res[1] = sum(x) - 1.0
        end

        n = 3
        optfunc = OptimizationFunction(robust_objective, OptimizationBase.AutoZygote();
                                      cons = robust_constraints)
        x0 = fill(1.0/n, n)
        prob = OptimizationProblem(optfunc, x0, nothing;
                                 lcons = [0.0],
                                 ucons = [Inf])

        sol = solve(prob, IpoptOptimizer())

        @test SciMLBase.successful_retcode(sol)
        @test sum(sol.u) >= 1.0 - 1e-6
    end

    # @testset "Complementarity Constraint" begin
    #     # Mathematical program with complementarity constraints (MPCC)
    #     # Reformulated using smoothing

    #     function mpcc_objective(x, p)
    #         return (x[1] - 1)^2 + (x[2] - 2)^2
    #     end

    #     function mpcc_constraints(res, x, p)
    #         # Original complementarity: x[1] * x[2] = 0
    #         # Smoothed version: x[1] * x[2] <= epsilon
    #         ε = 1e-6
    #         res[1] = x[1] * x[2] - ε
    #         # Additional constraint: x[1] + x[2] >= 1
    #         res[2] = x[1] + x[2] - 1.0
    #     end

    #     optfunc = OptimizationFunction(mpcc_objective, OptimizationBase.AutoZygote();
    #                                   cons = mpcc_constraints)
    #     x0 = [0.5, 0.5]
    #     prob = OptimizationProblem(optfunc, x0, nothing;
    #                              lb = [0.0, 0.0],
    #                              lcons = [-Inf, 0.0],
    #                              ucons = [0.0, Inf])

    #     sol = solve(prob, IpoptOptimizer())

    #     @test SciMLBase.successful_retcode(sol)
    #     # Should satisfy approximate complementarity
    #     @test sol.u[1] * sol.u[2] < 1e-4
    #     @test sol.u[1] + sol.u[2] >= 1.0 - 1e-6
    # end
end

@testset "Stress Tests" begin
    @testset "High-dimensional Problem" begin
        # Large-scale quadratic program
        n = 100

        # Random positive definite matrix
        A = randn(n, n)
        Q = A' * A + I
        b = randn(n)

        function large_quadratic(x, p)
            return 0.5 * dot(x, Q * x) - dot(b, x)
        end

        optfunc = OptimizationFunction(large_quadratic, OptimizationBase.AutoZygote())
        x0 = randn(n)
        prob = OptimizationProblem(optfunc, x0)

        sol = solve(prob, IpoptOptimizer();
                   maxiters = 1000)

        @test SciMLBase.successful_retcode(sol)
        # Check optimality: gradient should be near zero
        grad = Q * sol.u - b
        @test norm(grad) < 1e-4
    end

    @testset "Highly Nonlinear Problem" begin
        # Trigonometric test problem
        function trig_objective(x, p)
            n = length(x)
            return sum(sin(x[i])^2 * cos(x[i])^2 +
                      exp(-abs(x[i] - π/4)) for i in 1:n)
        end

        n = 10
        optfunc = OptimizationFunction(trig_objective, OptimizationBase.AutoZygote())
        x0 = randn(n)
        prob = OptimizationProblem(optfunc, x0;
                                 lb = fill(-2π, n),
                                 ub = fill(2π, n))

        sol = solve(prob, IpoptOptimizer(
                   hessian_approximation = "limited-memory"))

        @test SciMLBase.successful_retcode(sol)
    end
end
