using OptimizationBBO, Optimization, BlackBoxOptim
using Optimization.SciMLBase: MultiObjectiveOptimizationFunction
using Test

@testset "OptimizationBBO.jl" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 100.0]
    l1 = rosenbrock(x0, _p)

    optprob = OptimizationFunction(rosenbrock)
    prob = Optimization.OptimizationProblem(optprob, x0, _p, lb = [-1.0, -1.0],
        ub = [0.8, 0.8])
    sol = solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited())
    @test 10 * sol.objective < l1

    @test (@allocated solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited())) < 1e7

    prob = Optimization.OptimizationProblem(optprob, nothing, _p, lb = [-1.0, -1.0],
        ub = [0.8, 0.8])
    sol = solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited())
    @test 10 * sol.objective < l1

    sol = solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited(),
        callback = (args...) -> false)
    @test 10 * sol.objective < l1

    fitness_progress_history = []
    fitness_progress_history_orig = []
    loss_history = []
    function cb(state, fitness)
        push!(fitness_progress_history, state.objective)
        push!(fitness_progress_history_orig, BlackBoxOptim.best_fitness(state.original))
        push!(loss_history, fitness)
        return false
    end

    @testset "In-place Multi-Objective Optimization" begin
        function inplace_multi_obj!(cost, x, p)
            cost[1] = sum(x .^ 2)
            cost[2] = sum(x .^ 2 .- 10 .* cos.(2π .* x) .+ 10)
            return nothing
        end
        u0 = [0.25, 0.25]
        lb = [0.0, 0.0]
        ub = [2.0, 2.0]
        cost_prototype = zeros(2)
        mof_inplace = MultiObjectiveOptimizationFunction(inplace_multi_obj!; cost_prototype=cost_prototype)
        prob_inplace = Optimization.OptimizationProblem(mof_inplace, u0; lb=lb, ub=ub)
        sol_inplace = solve(prob_inplace, opt, NumDimensions=2, FitnessScheme=ParetoFitnessScheme{2}(is_minimizing=true))
        @test sol_inplace ≠ nothing
        @test length(sol_inplace.objective) == 2
        @test sol_inplace.objective[1] ≈ 6.9905986e-18 atol=1e-3
        @test sol_inplace.objective[2] ≈ 1.7763568e-15 atol=1e-3
    end

    @testset "Custom coalesce for Multi-Objective" begin
        function multi_obj_tuple(x, p)
            f1 = sum(x .^ 2)
            f2 = sum(x .^ 2 .- 10 .* cos.(2π .* x) .+ 10)
            return (f1, f2)
        end
        coalesce_sum(cost, x, p) = sum(cost)
        mof_coalesce = MultiObjectiveOptimizationFunction(multi_obj_tuple; coalesce=coalesce_sum)
        prob_coalesce = Optimization.OptimizationProblem(mof_coalesce, u0; lb=lb, ub=ub)
        sol_coalesce = solve(prob_coalesce, opt, NumDimensions=2, FitnessScheme=ParetoFitnessScheme{2}(is_minimizing=true))
        @test sol_coalesce ≠ nothing
        @test sol_coalesce.objective[1] ≈ 6.9905986e-18 atol=1e-3
        @test sol_coalesce.objective[2] ≈ 1.7763568e-15 atol=1e-3
        @test mof_coalesce.coalesce([1.0, 2.0], [0.0, 0.0], nothing) == 3.0
    end

    @testset "Error if in-place MultiObjectiveOptimizationFunction without cost_prototype" begin
        function inplace_multi_obj_err!(cost, x, p)
            cost[1] = sum(x .^ 2)
            cost[2] = sum(x .^ 2 .- 10 .* cos.(2π .* x) .+ 10)
            return nothing
        end
        @test_throws ArgumentError MultiObjectiveOptimizationFunction(inplace_multi_obj_err!)
    end
    sol = solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited(), callback = cb)
    # println(fitness_progress_history)
    @test !isempty(fitness_progress_history)
    fp1 = fitness_progress_history[1]
    fp2 = fitness_progress_history_orig[1]
    @test fp2 == fp1 == loss_history[1]

    @test_logs begin
        (Base.LogLevel(-1), "loss: 0.0")
        min_level = Base.LogLevel(-1)
        solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited(), progress = true)
    end

    @test_logs begin
        (Base.LogLevel(-1), "loss: 0.0")
        min_level = Base.LogLevel(-1)
        solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited(),
            progress = true,
            maxtime = 5)
    end

    # Define the initial guess and bounds
    u0 = [0.25, 0.25]
    lb = [0.0, 0.0]
    ub = [2.0, 2.0]

    # Define the optimizer
    opt = OptimizationBBO.BBO_borg_moea()

    @testset "Multi-Objective Optimization Tests" begin

        # Test 1: Sphere and Rastrigin Functions
        @testset "Sphere and Rastrigin Functions" begin
            function multi_obj_func_1(x, p)
                f1 = sum(x .^ 2)  # Sphere function
                f2 = sum(x .^ 2 .- 10 .* cos.(2π .* x) .+ 10)  # Rastrigin function
                return (f1, f2)
            end

            mof_1 = MultiObjectiveOptimizationFunction(multi_obj_func_1)
            prob_1 = Optimization.OptimizationProblem(mof_1, u0; lb = lb, ub = ub)
            sol_1 = solve(prob_1, opt, num_dimensions = 2,
                fitness_scheme = ParetoFitnessScheme{2}(is_minimizing = true))

            @test sol_1 ≠ nothing
            println("Solution for Sphere and Rastrigin: ", sol_1)
            @test sol_1.objective[1]≈6.9905986e-18 atol=1e-3
            @test sol_1.objective[2]≈1.7763568e-15 atol=1e-3
        end

        @testset "Sphere and Rastrigin Functions with callback" begin
            function multi_obj_func_1(x, p)
                f1 = sum(x .^ 2)  # Sphere function
                f2 = sum(x .^ 2 .- 10 .* cos.(2π .* x) .+ 10)  # Rastrigin function
                return (f1, f2)
            end

            fitness_progress_history = []
            fitness_progress_history_orig = []
            function cb(state, fitness)
                push!(fitness_progress_history, state.objective)
                push!(fitness_progress_history_orig,
                    BlackBoxOptim.best_fitness(state.original))
                return false
            end

            mof_1 = MultiObjectiveOptimizationFunction(multi_obj_func_1)
            prob_1 = Optimization.OptimizationProblem(mof_1, u0; lb = lb, ub = ub)
            sol_1 = solve(prob_1, opt, NumDimensions = 2,
                FitnessScheme = ParetoFitnessScheme{2}(is_minimizing = true),
                callback = cb)

            fp1 = fitness_progress_history[1]
            fp2 = fitness_progress_history_orig[1]
            @test fp2.orig == fp1
            @test length(fp1) == 2

            @test sol_1 ≠ nothing
            println("Solution for Sphere and Rastrigin: ", sol_1)
            @test sol_1.objective[1]≈6.9905986e-18 atol=1e-3
            @test sol_1.objective[2]≈1.7763568e-15 atol=1e-3
        end

        # Test 2: Rosenbrock and Ackley Functions
        @testset "Rosenbrock and Ackley Functions" begin
            function multi_obj_func_2(x, p)
                f1 = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2  # Rosenbrock function
                f2 = -20.0 * exp(-0.2 * sqrt(0.5 * (x[1]^2 + x[2]^2))) -
                     exp(0.5 * (cos(2π * x[1]) + cos(2π * x[2]))) + exp(1) + 20.0  # Ackley function
                return (f1, f2)
            end

            mof_2 = MultiObjectiveOptimizationFunction(multi_obj_func_2)
            prob_2 = Optimization.OptimizationProblem(mof_2, u0; lb = lb, ub = ub)
            sol_2 = solve(prob_2, opt, num_dimensions = 2,
                fitness_scheme = ParetoFitnessScheme{2}(is_minimizing = true))

            @test sol_2 ≠ nothing
            println("Solution for Rosenbrock and Ackley: ", sol_2)
            @test sol_2.objective[1]≈0.97438 atol=1e-3
            @test sol_2.objective[2]≈0.04088 atol=1e-3
        end

        # Test 3: ZDT1 Function
        @testset "ZDT1 Function" begin
            function multi_obj_func_3(x, p)
                f1 = x[1]
                g = 1 + 9 * sum(x[2:end]) / (length(x) - 1)
                f2 = g * (1 - sqrt(f1 / g))
                return (f1, f2)
            end

            mof_3 = MultiObjectiveOptimizationFunction(multi_obj_func_3)
            prob_3 = Optimization.OptimizationProblem(mof_3, u0; lb = lb, ub = ub)
            sol_3 = solve(prob_3, opt, num_dimensions = 2,
                fitness_scheme = ParetoFitnessScheme{2}(is_minimizing = true))

            @test sol_3 ≠ nothing
            println("Solution for ZDT1: ", sol_3)
            @test sol_3.objective[1]≈0.273445 atol=1e-3
            @test sol_3.objective[2]≈0.477079 atol=1e-3
        end
    end
end
