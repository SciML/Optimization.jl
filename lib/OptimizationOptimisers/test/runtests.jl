using OptimizationOptimisers, ForwardDiff, Optimization
using Test
using Zygote

@testset "OptimizationOptimisers.jl" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 100.0]
    l1 = rosenbrock(x0, _p)

    optprob = OptimizationFunction(rosenbrock, Optimization.AutoZygote())

    prob = OptimizationProblem(optprob, x0, _p)

    sol = Optimization.solve(prob,
        OptimizationOptimisers.Sophia(; η = 0.5,
            λ = 0.0),
        maxiters = 1000)
    @test 10 * sol.objective < l1

    prob = OptimizationProblem(optprob, x0, _p)
    sol = solve(prob, Optimisers.Adam(), maxiters = 1000, progress = false)
    @test 10 * sol.objective < l1

    x0 = 2 * ones(ComplexF64, 2)
    _p = ones(2)
    sumfunc(x0, _p) = sum(abs2, (x0 - _p))
    l1 = sumfunc(x0, _p)

    optprob = OptimizationFunction(sumfunc, Optimization.AutoZygote())

    prob = OptimizationProblem(optprob, x0, _p)

    sol = solve(prob, Optimisers.Adam(), maxiters = 1000)
    @test 10 * sol.objective < l1

    @testset "cache" begin
        objective(x, p) = (p[1] - x[1])^2
        x0 = zeros(1)
        p = [1.0]

        prob = OptimizationProblem(
            OptimizationFunction(objective,
                Optimization.AutoForwardDiff()), x0,
            p)
        cache = Optimization.init(prob, Optimisers.Adam(0.1), maxiters = 1000)
        sol = Optimization.solve!(cache)
        @test sol.u≈[1.0] atol=1e-3

        cache = Optimization.reinit!(cache; p = [2.0])
        sol = Optimization.solve!(cache)
        @test sol.u≈[2.0] atol=1e-3
    end

    @testset "callback" begin
        rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
        x0 = zeros(2)
        _p = [1.0, 100.0]
        l1 = rosenbrock(x0, _p)

        optprob = OptimizationFunction(rosenbrock, Optimization.AutoZygote())

        prob = OptimizationProblem(optprob, x0, _p)
        function callback(state, l)
            Optimisers.adjust!(state.original, 0.1 / state.iter)
            return false
        end
        sol = solve(prob,
            Optimisers.Adam(0.1),
            maxiters = 1000,
            progress = false,
            callback = callback)
    end

    @test_throws ArgumentError sol=solve(prob, Optimisers.Adam())
end
