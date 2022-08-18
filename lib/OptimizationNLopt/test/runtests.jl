using OptimizationNLopt, Optimization, Zygote
using Test

@testset "OptimizationNLopt.jl" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 100.0]
    l1 = rosenbrock(x0, _p)

    optprob = OptimizationFunction((x, p) -> -rosenbrock(x, p), Optimization.AutoZygote())
    prob = OptimizationProblem(optprob, x0, _p; sense = Optimization.MaxSense)
    sol = solve(prob, NLopt.Opt(:LN_BOBYQA, 2))
    @test 10 * sol.minimum < l1

    optprob = OptimizationFunction(rosenbrock, Optimization.AutoZygote())
    prob = OptimizationProblem(optprob, x0, _p)

    sol = solve(prob, NLopt.Opt(:LN_BOBYQA, 2))
    @test 10 * sol.minimum < l1

    sol = solve(prob, NLopt.Opt(:LD_LBFGS, 2))
    @test 10 * sol.minimum < l1

    prob = OptimizationProblem(optprob, x0, lb = [-1.0, -1.0], ub = [0.8, 0.8])
    sol = solve(prob, NLopt.Opt(:LD_LBFGS, 2))
    @test 10 * sol.minimum < l1

    sol = solve(prob, NLopt.Opt(:G_MLSL_LDS, 2), local_method = NLopt.Opt(:LD_LBFGS, 2),
                maxiters = 10000)
    @test 10 * sol.minimum < l1

    prob = OptimizationProblem(optprob, x0)
    sol = solve(prob, NLopt.LN_BOBYQA())
    @test 10 * sol.minimum < l1

    sol = solve(prob, NLopt.LD_LBFGS())
    @test 10 * sol.minimum < l1

    prob = OptimizationProblem(optprob, x0, lb = [-1.0, -1.0], ub = [0.8, 0.8])
    sol = solve(prob, NLopt.LD_LBFGS())
    @test 10 * sol.minimum < l1

    sol = solve(prob, NLopt.G_MLSL_LDS(), local_method = NLopt.LD_LBFGS(),
                local_maxiters = 10000, maxiters = 10000, population = 10)
    @test 10 * sol.minimum < l1
end
