using OptimizationOptimisers, ForwardDiff, Optimization
using Test
using Zygote
using Lux, MLUtils, Random, ComponentArrays, Printf, MLDataDevices

@testset "OptimizationOptimisers.jl" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 100.0]
    l1 = rosenbrock(x0, _p)

    optprob = OptimizationFunction(rosenbrock, Optimization.AutoZygote())

    prob = OptimizationProblem(optprob, x0, _p)

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
    @test sol.stats.iterations == 1000
    @test sol.stats.fevals == 1000
    @test sol.stats.gevals == 1000

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
        @test_broken sol.u≈[2.0] atol=1e-3
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

@testset "Minibatching" begin
    x = rand(Float32, 10000)
    y = sin.(x)
    data = MLUtils.DataLoader((x, y), batchsize = 100)

    # Define the neural network
    model = Chain(Dense(1, 32, tanh), Dense(32, 1))
    ps, st = Lux.setup(Random.default_rng(), model)
    ps_ca = ComponentArray(ps)
    smodel = StatefulLuxLayer{true}(model, nothing, st)

    function callback(state, l)
        state.iter % 25 == 1 && Printf.@printf "Iteration: %5d, Loss: %.6e\n" state.iter l
        return l < 1e-4
    end

    function loss(ps, data)
        ypred = [smodel([data[1][i]], ps)[1] for i in eachindex(data[1])]
        return sum(abs2, ypred .- data[2])
    end

    optf = OptimizationFunction(loss, AutoZygote())
    prob = OptimizationProblem(optf, ps_ca, data)

    res = Optimization.solve(prob, Optimisers.Adam(), epochs = 50)

    @test res.stats.iterations == 50*length(data)
    @test res.stats.fevals == 50*length(data)
    @test res.stats.gevals == 50*length(data)

    res = Optimization.solve(prob, Optimisers.Adam(), callback = callback, epochs = 100)

    @test res.objective < 1e-4

    data = CPUDevice()(data)
    optf = OptimizationFunction(loss, AutoZygote())
    prob = OptimizationProblem(optf, ps_ca, data)

    res = Optimization.solve(prob, Optimisers.Adam(), callback = callback, epochs = 10000)

    @test res.objective < 1e-4
end
