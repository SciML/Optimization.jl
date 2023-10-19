using OptimizationPRIMA, Optimization
using Test

@testset "OptimizationPRIMA.jl" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 100.0]
    l1 = rosenbrock(x0, _p)

    prob = OptimizationProblem(rosenbrock, x0, _p)
    sol = Optimization.solve(prob, UOBYQA(), maxiters = 1000)
    @test 10 * sol.objective < l1
    sol = Optimization.solve(prob, NEWUOA(), maxiters = 1000)
    @test 10 * sol.objective < l1
    sol = Optimization.solve(prob, BOBYQA(), maxiters = 1000)
    @test 10 * sol.objective < l1
    sol = Optimization.solve(prob, LINCOA(), maxiters = 1000)
    @test 10 * sol.objective < l1
    @test_throws SciMLBase.IncompatibleOptimizerError Optimization.solve(prob, COBYLA(), maxiters = 1000)

    function con2_c(res, x, p)
        res .= [x[1] + x[2], x[2] * sin(x[1]) - x[1]]
    end
    optprob = OptimizationFunction(rosenbrock, cons = con2_c)
    prob = OptimizationProblem(, x0, _p, lcons = [-Inf, -Inf], ucons = [Inf, Inf])
end
