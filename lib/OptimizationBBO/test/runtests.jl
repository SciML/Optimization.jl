using OptimizationBBO, Optimization
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
    function cb(state, fitness)
        push!(fitness_progress_history, [state.u, fitness])
        return false
    end
    sol = solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited(), callback = cb)
    # println(fitness_progress_history)
    @test !isempty(fitness_progress_history)

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
end
