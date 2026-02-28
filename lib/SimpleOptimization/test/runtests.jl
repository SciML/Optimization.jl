using SimpleOptimization, OptimizationBase
using Test

@testset "SimpleOptimization.jl" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = [0.5, 0.5]
    p = [1.0, 100.0]
    l1 = rosenbrock(x0, p)

    @testset "ForwardDiff" begin
        using ForwardDiff
        optf = OptimizationFunction(rosenbrock, OptimizationBase.AutoForwardDiff())
        prob = OptimizationProblem(optf, x0, p)

        sol = solve(prob, SimpleLBFGS())
        @test sol.objective < l1

        sol = solve(prob, SimpleBFGS())
        @test sol.objective < l1

        sol = solve(prob, SimpleGradientDescent(; eta = 0.001), maxiters = 10000)
        @test sol.objective < l1

        sol = solve(prob, SimpleNewton())
        @test sol.objective < l1
    end
end
