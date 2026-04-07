using OptimizationSOAP, OptimizationBase
using Test
using Zygote
using ForwardDiff

@testset "OptimizationSOAP.jl" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 100.0]
    l1 = rosenbrock(x0, _p)

    optprob = OptimizationFunction(rosenbrock, OptimizationBase.AutoZygote())
    prob = OptimizationProblem(optprob, x0, _p)

    sol = solve(prob, SOAP(eta = 0.003), maxiters = 1000, progress = false)
    @test 10 * sol.objective < l1

    optprob2 = OptimizationFunction(rosenbrock, OptimizationBase.AutoForwardDiff())
    prob2 = OptimizationProblem(optprob2, x0, _p)

    sol2 = solve(prob2, SOAP(eta = 0.003), maxiters = 1000)
    @test 10 * sol2.objective < l1
    @test sol2.stats.iterations == 1000
    @test sol2.stats.fevals == 1000
    @test sol2.stats.gevals == 1000

    @testset "epochs & maxiters" begin
        using MLUtils
        optf = OptimizationFunction(
            (u, data) -> sum(u) + sum(data), OptimizationBase.AutoZygote()
        )
        prob = OptimizationProblem(optf, ones(2), MLUtils.DataLoader(ones(2, 2)))
        @test_throws ArgumentError solve(prob, SOAP())
        @test_throws ArgumentError solve(prob, SOAP(), epochs = 2, maxiters = 2)
        sol = solve(prob, SOAP(), epochs = 2)
        @test sol.stats.iterations == 4
        sol = solve(prob, SOAP(), maxiters = 2)
        @test sol.stats.iterations == 2
    end

    @testset "callback" begin
        called = Ref(0)
        function cb(state, l)
            called[] += 1
            return false
        end
        sol = solve(prob, SOAP(eta = 0.003), maxiters = 100, callback = cb)
        @test called[] >= 100
    end

    @testset "early stopping" begin
        sol = solve(prob, SOAP(eta = 0.003), maxiters = 1000,
            callback = (state, l) -> state.iter >= 50)
        @test sol.stats.iterations == 50
    end

    @testset "Optimisers.jl interface" begin
        opt = SOAP(eta = 0.01, freq = 3)

        W = randn(Float32, 4, 8)
        st = Optimisers.setup(opt, W)
        for i in 1:6
            g = randn(Float32, 4, 8) .* 0.1f0
            st, W = Optimisers.update(st, W, g)
        end
        @test size(W) == (4, 8)
        @test all(isfinite, W)

        b = randn(Float32, 4)
        st_b = Optimisers.setup(opt, b)
        for i in 1:5
            g = randn(Float32, 4) .* 0.1f0
            st_b, b = Optimisers.update(st_b, b, g)
        end
        @test length(b) == 4
        @test all(isfinite, b)
    end
end