using OptimizationNOMAD, OptimizationBase
using Test

@testset "OptimizationNOMAD.jl" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 100.0]
    l1 = rosenbrock(x0, _p)

    f = OptimizationFunction(rosenbrock)

    prob = OptimizationProblem(f, x0, _p)
    sol = OptimizationBase.solve(prob, NOMADOpt())
    @test 10 * sol.objective < l1

    prob = OptimizationProblem(f, x0, _p; lb = [-1.0, -1.0], ub = [1.5, 1.5])
    sol = OptimizationBase.solve(prob, NOMADOpt())
    @test 10 * sol.objective < l1

    cons = (res, x, p) -> (res[1] = x[1]^2 + x[2]^2; nothing)
    f = OptimizationFunction(rosenbrock, cons = cons)
    prob = OptimizationProblem(f, x0, _p; lcons = [-Inf], ucons = [1.0])
    sol = OptimizationBase.solve(prob, NOMADOpt(), maxiters = 5000)
    @test 10 * sol.objective < l1

    function con2_c(res, x, p)
        res .= [x[1]^2 + x[2]^2, x[2] * sin(x[1]) - x[1]]
    end

    f = OptimizationFunction(rosenbrock, cons = con2_c)
    prob = OptimizationProblem(f, x0, _p; lcons = [-Inf, -Inf], ucons = [0.5, 0.0])
    sol = OptimizationBase.solve(prob, NOMADOpt(), maxiters = 5000)
    @test sol.objective < l1
end
