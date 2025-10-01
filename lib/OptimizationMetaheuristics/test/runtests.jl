using OptimizationMetaheuristics, Optimization, Random
using Test

Random.seed!(42)
@testset "OptimizationMetaheuristics.jl" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 100.0]
    l1 = rosenbrock(x0, _p)
    optprob = OptimizationFunction(rosenbrock)
    prob = Optimization.OptimizationProblem(optprob, x0, _p, lb = [-1.0, -1.0],
        ub = [1.0, 1.0])
    sol = solve(prob, ECA())
    @test 10 * sol.objective < l1

    sol = solve(prob, Metaheuristics.DE())
    @test 10 * sol.objective < l1

    sol = solve(prob, PSO())
    @test 10 * sol.objective < l1

    sol = solve(prob, ABC())
    @test 10 * sol.objective < l1

    sol = solve(prob, CGSA(N = 100))
    @test 10 * sol.objective < l1

    sol = solve(prob, SA())
    @test 10 * sol.objective < l1

    sol = solve(prob, WOA())
    @test 10 * sol.objective < l1

    sol = solve(prob, ECA())
    @test 10 * sol.objective < l1

    sol = solve(prob, Metaheuristics.DE(), use_initial = true)
    @test 10 * sol.objective < l1

    sol = solve(prob, PSO(), use_initial = true)
    @test 10 * sol.objective < l1

    sol = solve(prob, ABC(), use_initial = true)
    @test 10 * sol.objective < l1

    sol = solve(prob, CGSA(N = 100), use_initial = true)
    @test 10 * sol.objective < l1

    sol = solve(prob, SA(), use_initial = true)
    @test 10 * sol.objective < l1

    sol = solve(prob, WOA(), use_initial = true)
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
            (schaffer_n2, [0.0, 0.0, 0.0], [2.0, 0.0, 0.0])
        ]

        nobjectives = 2
        npartitions = 100

        # Define the different algorithms
        algs = [
            NSGA2(),
            NSGA3(),
            SPEA2(),
            CCMO(NSGA2(N = 100, p_m = 0.001)),
            MOEAD_DE(gen_ref_dirs(nobjectives, npartitions),
                options = Options(debug = false, iterations = 10000)),
            SMS_EMOA()
        ]
        Random.seed!(42)
        # Run tests for each problem and algorithm
        for (prob_func, lb, ub) in problems
            prob_name = string(prob_func)
            for alg in algs
                alg_name = string(typeof(alg))
                @testset "$alg_name on $prob_name" begin
                    multi_obj_fun = MultiObjectiveOptimizationFunction((
                        x, p) -> prob_func(x))
                    prob = OptimizationProblem(multi_obj_fun, lb; lb = lb, ub = ub)
                    if (alg_name == "Metaheuristics.Algorithm{CCMO{NSGA2}}")
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
end
