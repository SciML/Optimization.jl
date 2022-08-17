using OptimizationMetaheuristics, Optimization
using Test

@testset "OptimizationMetaheuristics.jl" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 100.0]
    l1 = rosenbrock(x0, _p)
    optprob = OptimizationFunction(rosenbrock)
    prob = Optimization.OptimizationProblem(optprob, x0, _p, lb = [-1.0, -1.0],
                                            ub = [1.5, 1.5])
    sol = solve(prob, ECA())
    @test 10 * sol.minimum < l1

    sol = solve(prob, Metaheuristics.DE())
    @test 10 * sol.minimum < l1

    sol = solve(prob, PSO())
    @test 10 * sol.minimum < l1

    sol = solve(prob, ABC())
    @test 10 * sol.minimum < l1

    sol = solve(prob, CGSA(N = 100))
    @test 10 * sol.minimum < l1

    sol = solve(prob, SA())
    @test 10 * sol.minimum < l1

    sol = solve(prob, WOA())
    @test 10 * sol.minimum < l1

    sol = solve(prob, ECA())
    @test 10 * sol.minimum < l1

    sol = solve(prob, Metaheuristics.DE(), use_initial = true)
    @test 10 * sol.minimum < l1

    sol = solve(prob, PSO(), use_initial = true)
    @test 10 * sol.minimum < l1

    sol = solve(prob, ABC(), use_initial = true)
    @test 10 * sol.minimum < l1

    sol = solve(prob, CGSA(N = 100), use_initial = true)
    @test 10 * sol.minimum < l1

    sol = solve(prob, SA(), use_initial = true)
    @test 10 * sol.minimum < l1

    sol = solve(prob, WOA(), use_initial = true)
    @test 10 * sol.minimum < l1
end
