using OptimizationMetaheuristics, OptimizationBase, Random
using Test

Random.seed!(42)
@testset "OptimizationMetaheuristics.jl" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 100.0]
    l1 = rosenbrock(x0, _p)
    optprob = OptimizationFunction(rosenbrock)
    prob = OptimizationBase.OptimizationProblem(
        optprob, x0, _p, lb = [-1.0, -1.0],
        ub = [1.0, 1.0]
    )
    sol = solve(prob, Metaheuristics.ECA())
    @test 10 * sol.objective < l1

    sol = solve(prob, Metaheuristics.DE())
    @test 10 * sol.objective < l1

    sol = solve(prob, Metaheuristics.PSO())
    @test 10 * sol.objective < l1

    sol = solve(prob, Metaheuristics.ABC())
    @test 10 * sol.objective < l1

    sol = solve(prob, Metaheuristics.CGSA(N = 100))
    @test 10 * sol.objective < l1

    sol = solve(prob, Metaheuristics.SA())
    @test 10 * sol.objective < l1

    sol = solve(prob, Metaheuristics.WOA())
    @test 10 * sol.objective < l1

    sol = solve(prob, Metaheuristics.ECA())
    @test 10 * sol.objective < l1

    sol = solve(prob, Metaheuristics.DE(), use_initial = true)
    @test 10 * sol.objective < l1

    sol = solve(prob, Metaheuristics.PSO(), use_initial = true)
    @test 10 * sol.objective < l1

    sol = solve(prob, Metaheuristics.ABC(), use_initial = true)
    @test 10 * sol.objective < l1

    sol = solve(prob, Metaheuristics.CGSA(N = 100), use_initial = true)
    @test 10 * sol.objective < l1

    sol = solve(prob, Metaheuristics.SA(), use_initial = true)
    @test 10 * sol.objective < l1

    sol = solve(prob, Metaheuristics.WOA(), use_initial = true)
    @test 10 * sol.objective < l1

    # Define the benchmark functions as multi-objective problems
    function sphere(x)
        f1 = sum(x .^ 2)
        f2 = sum((x .- 2.0) .^ 2)
        gx = [0.0]
        hx = [0.0]
        return [f1, f2], gx, hx
    end

    function rastrigin(x)
        f1 = sum(x .^ 2 .- 10 .* cos.(2 .* π .* x) .+ 10)
        f2 = sum((x .- 2.0) .^ 2 .- 10 .* cos.(2 .* π .* (x .- 2.0)) .+ 10)
        gx = [0.0]
        hx = [0.0]
        return [f1, f2], gx, hx
    end

    function rosenbrock(x)
        f1 = sum(100 .* (x[2:end] .- x[1:(end - 1)] .^ 2) .^ 2 .+ (x[1:(end - 1)] .- 1) .^ 2)
        f2 = sum(100 .* ((x[2:end] .- 2.0) .- (x[1:(end - 1)] .^ 2)) .^ 2 .+ ((x[1:(end - 1)] .- 1.0) .^ 2))
        gx = [0.0]
        hx = [0.0]
        return [f1, f2], gx, hx
    end

    function ackley(x)
        f1 = -20 * exp(-0.2 * sqrt(sum(x .^ 2) / length(x))) -
            exp(sum(cos.(2 * π .* x)) / length(x)) + 20 + ℯ
        f2 = -20 * exp(-0.2 * sqrt(sum((x .- 2.0) .^ 2) / length(x))) -
            exp(sum(cos.(2 * π .* (x .- 2.0))) / length(x)) + 20 + ℯ
        gx = [0.0]
        hx = [0.0]
        return [f1, f2], gx, hx
    end

    function dtlz2(x)
        g = sum((x[3:end] .- 0.5) .^ 2)
        f1 = (1 + g) * cos(x[1] * π / 2) * cos(x[2] * π / 2)
        f2 = (1 + g) * cos(x[1] * π / 2) * sin(x[2] * π / 2)
        gx = [0.0]
        hx = [0.0]
        return [f1, f2], gx, hx
    end

    function schaffer_n2(x)
        f1 = x[1]^2
        f2 = (x[1] - 2.0)^2
        gx = [0.0]
        hx = [0.0]
        return [f1, f2], gx, hx
    end
    # Define the testset
    @testset "Multi-Objective Optimization with Various Functions and Metaheuristics" begin
        # Define the problems and their bounds
        problems = [
            (sphere, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            (rastrigin, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            (rosenbrock, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            (ackley, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            (dtlz2, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            (schaffer_n2, [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]),
        ]

        nobjectives = 2
        npartitions = 100

        # Define the different algorithms
        algs = [
            Metaheuristics.NSGA2(),
            Metaheuristics.NSGA3(),
            Metaheuristics.SPEA2(),
            Metaheuristics.CCMO(Metaheuristics.NSGA2(N = 100, p_m = 0.001)),
            Metaheuristics.MOEAD_DE(
                Metaheuristics.gen_ref_dirs(nobjectives, npartitions),
                options = Metaheuristics.Options(debug = false, iterations = 10000)
            ),
            Metaheuristics.SMS_EMOA(),
        ]
        Random.seed!(42)
        # Run tests for each problem and algorithm
        for (prob_func, lb, ub) in problems
            prob_name = string(prob_func)
            for alg in algs
                alg_name = string(typeof(alg))
                @testset "$alg_name on $prob_name" begin
                    multi_obj_fun = MultiObjectiveOptimizationFunction(
                        (
                            x, p,
                        ) -> prob_func(x)
                    )
                    prob = OptimizationProblem(multi_obj_fun, lb; lb = lb, ub = ub)
                    # CCMO wraps a base algorithm and has no `N` on its own
                    # parameters, so the `use_initial` seeding path does not apply.
                    # Detect it by type rather than a stringified type name, which
                    # is fragile across Metaheuristics' type-printing changes.
                    if alg.parameters isa Metaheuristics.CCMO
                        sol = solve(prob, alg)
                    else
                        sol = solve(prob, alg; maxiters = 10000, use_initial = true)
                    end

                    # Tests
                    @test !isempty(sol.u)  # Check that a solution was found
                end
            end
        end
    end

    @testset "Nonlinear constraints" begin
        rb(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
        p = [1.0, 100.0]
        # The unconstrained minimizer [1, 1] is outside the disk, so the constrained minimizer
        # is on its boundary.
        r = sqrt(0.5)
        f_ref = minimum(rb(r .* [cos(φ), sin(φ)], p) for φ in range(0, 2π, length = 100_000))
        disk(res, x, p) = (res .= [x[1]^2 + x[2]^2])
        optf = OptimizationFunction(rb; cons = disk)
        prob = OptimizationProblem(
            optf, zeros(2), p; lb = [-1.0, -1.0], ub = [1.0, 1.0],
            lcons = [-Inf], ucons = [0.5]
        )
        @testset "Upper bound with $(nameof(typeof(alg).parameters[1]))" for alg in (
                DE(), PSO(), SHADE(), εDE(),
            )
            Random.seed!(1)
            sol = solve(prob, alg)
            @test sol.u[1]^2 + sol.u[2]^2 <= 0.5
            @test sol.objective <= f_ref + 1.0e-2
        end
        # ECA terminates before reaching the minimizer of this problem also when called
        # through Metaheuristics.optimize directly, so only feasibility is tested.
        Random.seed!(1)
        sol = solve(prob, ECA())
        @test sol.u[1]^2 + sol.u[2]^2 <= 0.5

        # A lower bound excluding the unconstrained minimizer and an upper bound in a
        # second constraint
        Random.seed!(1)
        cons2(res, x, p) = (res .= [x[1]^2 + x[2]^2, x[1]])
        prob = OptimizationProblem(
            OptimizationFunction(rb; cons = cons2), zeros(2), p;
            lb = [-2.0, -2.0], ub = [2.0, 2.0], lcons = [2.5, -Inf], ucons = [Inf, 1.0]
        )
        sol = solve(prob, DE())
        @test sol.u[1]^2 + sol.u[2]^2 >= 2.5
        @test sol.u[1] <= 1.0

        # Equality constraint, satisfied to within the tolerance `h_tol`
        Random.seed!(1)
        line(res, x, p) = (res .= [x[1] + x[2]])
        prob = OptimizationProblem(
            OptimizationFunction(rb; cons = line), zeros(2), p;
            lb = [-2.0, -2.0], ub = [2.0, 2.0], lcons = [1.0], ucons = [1.0]
        )
        sol = solve(prob, DE(options = Metaheuristics.Options(h_tol = 1.0e-4)))
        @test abs(sol.u[1] + sol.u[2] - 1) <= 1.0e-4 + 1.0e-12
        # The minimizer of rb on the line x[2] = 1 - x[1]
        φ = range(-2, 2, length = 400_001)
        @test sol.objective <= minimum(x -> rb([x, 1 - x], p), φ) + 1.0e-2

        # Algorithms without constraint handling reject constrained problems
        prob = OptimizationProblem(
            optf, zeros(2), p; lb = [-1.0, -1.0], ub = [1.0, 1.0],
            lcons = [-Inf], ucons = [0.5]
        )
        @test_throws OptimizationBase.IncompatibleOptimizerError solve(prob, ABC())

        # Constraints of the problem are appended to those returned by a multi-objective
        # function
        Random.seed!(1)
        biobj(x, p) = ([sum(abs2, x), sum(abs2, x .- 2)], [0.0], [0.0])
        mof = MultiObjectiveOptimizationFunction(biobj; cons = (res, x, p) -> (res .= [x[1]]))
        prob = OptimizationProblem(
            mof, zeros(2), nothing; lb = [-1.0, -1.0], ub = [3.0, 3.0],
            lcons = [-Inf], ucons = [0.5]
        )
        sol = solve(prob, NSGA2())
        @test all(<=(0.5), Metaheuristics.positions(sol.original)[:, 1])
    end

    # Default callback must be a stable const so init with/without kwargs share a cache type.
    @testset "DEFAULT_CALLBACK type stability across kwargs" begin
        rb(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
        optf = OptimizationFunction(rb)
        prob = OptimizationProblem(
            optf, zeros(2), [1.0, 100.0], lb = [-1.0, -1.0], ub = [1.0, 1.0]
        )
        @test typeof(OptimizationBase.init(prob, Metaheuristics.ECA())) ==
            typeof(OptimizationBase.init(prob, Metaheuristics.ECA(); maxiters = 100))
    end
end
