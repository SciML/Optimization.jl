using OptimizationPRIMA, OptimizationBase, ForwardDiff, ModelingToolkit, ReverseDiff
using Test

@testset "OptimizationPRIMA.jl" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 100.0]
    l1 = rosenbrock(x0, _p)

    prob = OptimizationProblem(rosenbrock, x0, _p)
    sol = OptimizationBase.solve(prob, UOBYQA(), maxiters = 1000)
    @test 10 * sol.objective < l1
    sol = OptimizationBase.solve(prob, NEWUOA(), maxiters = 1000)
    @test 10 * sol.objective < l1
    sol = OptimizationBase.solve(prob, BOBYQA(), maxiters = 1000)
    @test 10 * sol.objective < l1
    sol = OptimizationBase.solve(prob, LINCOA(), maxiters = 1000)
    @test 10 * sol.objective < l1
    @test_throws OptimizationBase.IncompatibleOptimizerError OptimizationBase.solve(
        prob, COBYLA(), maxiters = 1000)

    function con2_c(res, x, p)
        res .= [x[1] + x[2], x[2] * sin(x[1]) - x[1]]
    end
    optprob = OptimizationFunction(rosenbrock, AutoForwardDiff(), cons = con2_c)
    prob = OptimizationProblem(optprob, x0, _p, lcons = [1, -100], ucons = [1, 100])
    sol = OptimizationBase.solve(prob, COBYLA(), maxiters = 1000)
    @test sol.objective < l1

    function con2_c(res, x, p)
        res .= [x[1] + x[2]]
    end
    optprob = OptimizationFunction(rosenbrock, AutoForwardDiff(), cons = con2_c)
    prob = OptimizationProblem(optprob, x0, _p, lcons = [1], ucons = [1])
    sol = OptimizationBase.solve(prob, COBYLA(), maxiters = 1000)
    @test sol.objective < l1

    prob = OptimizationProblem(optprob, x0, _p, lcons = [1], ucons = [5])
    sol = OptimizationBase.solve(prob, COBYLA(), maxiters = 1000)
    @test sol.objective < l1

    function con2_c(res, x, p)
        res .= [x[2] * sin(x[1]) - x[1]]
    end
    optprob = OptimizationFunction(rosenbrock, AutoSymbolics(), cons = con2_c)
    prob = OptimizationProblem(optprob, x0, _p, lcons = [10], ucons = [50])
    sol = OptimizationBase.solve(prob, COBYLA(), maxiters = 1000)
    @test 10 * sol.objective < l1
end
