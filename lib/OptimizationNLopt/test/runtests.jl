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
    @test sol.retcode == ReturnCode.Success
    @test 10 * sol.objective < l1

    optprob = OptimizationFunction(rosenbrock, Optimization.AutoZygote())
    prob = OptimizationProblem(optprob, x0, _p)

    sol = solve(prob, NLopt.Opt(:LN_BOBYQA, 2))
    @test sol.retcode == ReturnCode.Success
    @test 10 * sol.objective < l1

    sol = solve(prob, NLopt.Opt(:LD_LBFGS, 2))
    @test sol.retcode == ReturnCode.Success
    @test 10 * sol.objective < l1

    prob = OptimizationProblem(optprob, x0, lb = [-1.0, -1.0], ub = [0.8, 0.8])
    sol = solve(prob, NLopt.Opt(:LD_LBFGS, 2))
    @test sol.retcode == ReturnCode.Success
    @test 10 * sol.objective < l1

    sol = solve(prob, NLopt.Opt(:G_MLSL_LDS, 2), local_method = NLopt.Opt(:LD_LBFGS, 2),
                maxiters = 10000)
    @test sol.retcode == ReturnCode.Success
    @test 10 * sol.objective < l1

    prob = OptimizationProblem(optprob, x0)
    sol = solve(prob, NLopt.LN_BOBYQA())
    @test sol.retcode == ReturnCode.Success
    @test 10 * sol.objective < l1

    sol = solve(prob, NLopt.LD_LBFGS())
    @test sol.retcode == ReturnCode.Success
    @test 10 * sol.objective < l1

    prob = OptimizationProblem(optprob, x0, lb = [-1.0, -1.0], ub = [0.8, 0.8])
    sol = solve(prob, NLopt.LD_LBFGS())
    @test sol.retcode == ReturnCode.Success
    @test 10 * sol.objective < l1

    sol = solve(prob, NLopt.G_MLSL_LDS(), local_method = NLopt.LD_LBFGS(),
                local_maxiters = 10000, maxiters = 10000, population = 10)
    @test sol.retcode == ReturnCode.Success
    @test 10 * sol.objective < l1

    @testset "cache" begin
        objective(x, p) = (p[1] - x[1])^2
        x0 = zeros(1)
        p = [1.0]

        prob = OptimizationProblem(objective, x0, p)
    cache = Optimization.init(prob, NLopt.Opt(:LD_LBFGS, 1))
    sol = Optimization.solve!(cache)
    @test sol.retcode == ReturnCode.Success
    @test sol.u ≈ [1.0]

    cache = Optimization.reinit!(cache; p = [2.0])
    sol = Optimization.solve!(cache)
    @test sol.retcode == ReturnCode.Success
    @test sol.u ≈ [2.0]
    end
end
