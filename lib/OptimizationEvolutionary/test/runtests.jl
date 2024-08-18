using OptimizationEvolutionary, Optimization, Random
using Optimization.SciMLBase: MultiObjectiveOptimizationFunction
using Test

Random.seed!(1234)
@testset "OptimizationEvolutionary.jl" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 100.0]
    l1 = rosenbrock(x0, _p)
    optprob = OptimizationFunction(rosenbrock)
    prob = Optimization.OptimizationProblem(optprob, x0, _p)
    sol = solve(prob, CMAES(μ = 40, λ = 100), abstol = 1e-15)
    @test 10 * sol.objective < l1

    x0 = [-0.7, 0.3]
    prob = Optimization.OptimizationProblem(optprob, x0, _p, lb = [0.0, 0.0],
        ub = [0.5, 0.5])
    sol = solve(prob, CMAES(μ = 50, λ = 60))
    @test sol.u == zeros(2)

    x0 = zeros(2)
    cons_circ = (res, x, p) -> res .= [x[1]^2 + x[2]^2]
    optprob = OptimizationFunction(rosenbrock; cons = cons_circ)
    prob = OptimizationProblem(optprob, x0, _p, lcons = [-Inf], ucons = [0.25^2])
    sol = solve(prob, CMAES(μ = 40, λ = 100))
    res = zeros(1)
    cons_circ(res, sol.u, nothing)
    @test res[1]≈0.0625 atol=1e-5
    @test sol.objective < l1

    prob = OptimizationProblem(optprob, x0, _p, lcons = [-Inf], ucons = [5.0],
        lb = [0.0, 1.0], ub = [Inf, Inf])
    sol = solve(prob, CMAES(μ = 40, λ = 100))
    res = zeros(1)
    cons_circ(res, sol.u, nothing)
    @test sol.objective < l1

    function cb(state, args...)
        if state.iter % 10 == 0
            println(state.u)
        end
        return false
    end
    solve(prob, CMAES(μ = 40, λ = 100), callback = cb, maxiters = 100)

    # Test compatibility of user overload of trace! 
    function Evolutionary.trace!(
            record::Dict{String, Any}, objfun, state, population, method::CMAES, options)
        # record fittest individual
        record["TESTVAL"] = state.fittest
    end

    # Test that `store_trace=true` works now. Threw ""type Array has no field value" before.
    sol = solve(prob, CMAES(μ = 40, λ = 100), store_trace = true)

    # Make sure that both the user's trace record value, as well as `curr_u` are stored in the trace.
    @test haskey(sol.original.trace[end].metadata, "TESTVAL") &&
          haskey(sol.original.trace[end].metadata, "curr_u")

    # Test Suite for Different Multi-Objective Functions
function test_multi_objective(func, initial_guess)
    # Define the gradient function using ForwardDiff
    function gradient_multi_objective(x, p=nothing)
        ForwardDiff.jacobian(func, x)
    end

    # Create an instance of MultiObjectiveOptimizationFunction
    obj_func = MultiObjectiveOptimizationFunction(func, jac=gradient_multi_objective)

    # Set up the evolutionary algorithm (e.g., NSGA2)
    algorithm = OptimizationEvolutionary.NSGA2()

    # Define the optimization problem
    problem = OptimizationProblem(obj_func, initial_guess)

    # Solve the optimization problem
    result = solve(problem, algorithm)
    
    return result
end

@testset "Multi-Objective Optimization Tests" begin

    # Test 1: Sphere and Rastrigin Functions
    @testset "Sphere and Rastrigin Functions" begin
        function multi_objective_1(x, p=nothing)::Vector{Float64}
            f1 = sum(x .^ 2)  # Sphere function
            f2 = sum(x .^ 2 .- 10 .* cos.(2π .* x) .+ 10)  # Rastrigin function
            return [f1, f2]
        end
        result = test_multi_objective(multi_objective_1, [0.0, 1.0])
        @test result ≠ nothing
        println("Solution for Sphere and Rastrigin: ", result)
        @test result.u[1][1] ≈ 7.88866e-5 atol=1e-3
        @test result.u[1][2] ≈ 4.96471e-5 atol=1e-3
        @test result.objective[1] ≈ 8.6879e-9 atol=1e-3
    end

    # Test 2: Rosenbrock and Ackley Functions
    @testset "Rosenbrock and Ackley Functions" begin
        function multi_objective_2(x, p=nothing)::Vector{Float64}
            f1 = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2  # Rosenbrock function
            f2 = -20.0 * exp(-0.2 * sqrt(0.5 * (x[1]^2 + x[2]^2))) - exp(0.5 * (cos(2π * x[1]) + cos(2π * x[2]))) + exp(1) + 20.0  # Ackley function
            return [f1, f2]
        end
        result = test_multi_objective(multi_objective_2, [1.0, 1.0])
        @test result ≠ nothing
        println("Solution for Rosenbrock and Ackley: ", result)
        @test result.u[10][1] ≈ 1.0 atol=1e-3
        @test result.u[10][2] ≈ 0.999739 atol=1e-3
        @test result.objective[2] ≈ 3.625384 atol=1e-3
    end

    # Test 3: ZDT1 Function
    @testset "ZDT1 Function" begin
        function multi_objective_3(x, p=nothing)::Vector{Float64}
            f1 = x[1]
            g = 1 + 9 * sum(x[2:end]) / (length(x) - 1)
            sqrt_arg = f1 / g
            f2 = g * (1 - (sqrt_arg >= 0 ? sqrt(sqrt_arg) : NaN))
            return [f1, f2]
        end
        result = test_multi_objective(multi_objective_3, [0.25, 1.5])
        @test result ≠ nothing
        println("Solution for ZDT1: ", result)
        @test result.u[1][1] ≈ -0.365434 atol=1e-3
        @test result.u[1][2] ≈ 1.22128 atol=1e-3
        @test result.objective[1] ≈ -0.365434 atol=1e-3
    end

    # Test 4: DTLZ2 Function
    @testset "DTLZ2 Function" begin
        function multi_objective_4(x, p=nothing)::Vector{Float64}
            f1 = (1 + sum(x[2:end] .^ 2)) * cos(x[1] * π / 2)
            f2 = (1 + sum(x[2:end] .^ 2)) * sin(x[1] * π / 2)
            return [f1, f2]
        end
        result = test_multi_objective(multi_objective_4, [0.25, 0.75])
        @test result ≠ nothing
        println("Solution for DTLZ2: ", result)
        @test result.u[1][1] ≈ 0.899183 atol=1e-3
        @test result.u[2][1] ≈ 0.713992 atol=1e-3
        @test result.objective[1] ≈ 0.1599915 atol=1e-3
    end

    # Test 5: Schaffer Function N.2
    @testset "Schaffer Function N.2" begin
        function multi_objective_5(x, p=nothing)::Vector{Float64}
            f1 = x[1]^2
            f2 = (x[1] - 2)^2
            return [f1, f2]
        end
        result = test_multi_objective(multi_objective_5, [1.0])
        @test result ≠ nothing
        println("Solution for Schaffer N.2: ", result)
        @test result.u[19][1] ≈ 0.252635 atol=1e-3
        @test result.u[9][1] ≈ 1.0 atol=1e-3
        @test result.objective[1] ≈ 1.0 atol=1e-3
    end

end
end
