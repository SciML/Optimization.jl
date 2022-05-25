using GalacticOptimJL, GalacticOptimJL.Optim, GalacticOptim, ForwardDiff, Zygote, Random, ModelingToolkit
using Test

@testset "GalacticOptimJL.jl" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 100.0]
    l1 = rosenbrock(x0, _p)
    f = OptimizationFunction(rosenbrock, GalacticOptim.AutoForwardDiff())
    prob = OptimizationProblem(f, x0, _p)
    Random.seed!(1234)
    sol = solve(prob, SimulatedAnnealing())
    @test 10 * sol.minimum < l1

    Random.seed!(1234)
    prob = OptimizationProblem(f, x0, _p, lb=[-1.0, -1.0], ub=[0.8, 0.8])
    sol = solve(prob, SAMIN())
    @test 10 * sol.minimum < l1

    prob = OptimizationProblem(rosenbrock, x0, _p)
    sol = solve(prob, Optim.NelderMead(;initial_simplex=Optim.AffineSimplexer(;a = 0.025, b = 0.5)))
    @test 10 * sol.minimum < l1

    cons = (x, p) -> [x[1]^2 + x[2]^2]
    optprob = OptimizationFunction(rosenbrock, GalacticOptim.AutoForwardDiff(); cons=cons)
    optprob = OptimizationFunction(rosenbrock, GalacticOptim.AutoModelingToolkit(); cons=cons)

    prob = OptimizationProblem(optprob, x0, _p)

    sol = solve(prob, Optim.BFGS())
    @test 10 * sol.minimum < l1

    sol = solve(prob, Optim.Newton())
    @test 10 * sol.minimum < l1

    sol = solve(prob, Optim.KrylovTrustRegion())
    @test 10 * sol.minimum < l1

    prob = OptimizationProblem(optprob, x0, _p, lcons=[-Inf], ucons=[Inf])
    sol = solve(prob, IPNewton())
    @test 10 * sol.minimum < l1

    prob = OptimizationProblem(optprob, x0, _p, lcons=[-5.0], ucons=[10.0])
    sol = solve(prob, IPNewton())
    @test 10 * sol.minimum < l1

    prob = OptimizationProblem(optprob, x0, _p, lcons=[-Inf], ucons=[Inf], lb=[-500.0, -500.0], ub=[50.0, 50.0])
    sol = solve(prob, IPNewton())
    @test sol.minimum < l1

    function con2_c(x, p)
        [x[1]^2 + x[2]^2, x[2] * sin(x[1]) - x[1]]
    end

    optprob = OptimizationFunction(rosenbrock, GalacticOptim.AutoForwardDiff(); cons=con2_c)
    prob = OptimizationProblem(optprob, x0, _p, lcons=[-Inf, -Inf], ucons=[Inf, Inf])
    sol = solve(prob, IPNewton())
    @test 10 * sol.minimum < l1

    cons_circ = (x, p) -> [x[1]^2 + x[2]^2]
    optprob = OptimizationFunction(rosenbrock, GalacticOptim.AutoForwardDiff(); cons=cons_circ)
    prob = OptimizationProblem(optprob, x0, _p, lcons=[-Inf], ucons=[0.25^2])
    sol = solve(prob, IPNewton())
    @test sqrt(cons(sol.u, nothing)[1]) ≈ 0.25 rtol = 1e-6

    optprob = OptimizationFunction(rosenbrock, GalacticOptim.AutoZygote())

    prob = OptimizationProblem(optprob, x0, _p, lb=[-1.0, -1.0], ub=[0.8, 0.8])
    sol = solve(prob, Optim.Fminbox())
    @test 10 * sol.minimum < l1

    Random.seed!(1234)
    prob = OptimizationProblem(optprob, x0, _p, lb=[-1.0, -1.0], ub=[0.8, 0.8])
    sol = solve(prob, Optim.SAMIN())
    @test 10 * sol.minimum < l1

    optprob = OptimizationFunction((x, p) -> -rosenbrock(x, p), GalacticOptim.AutoZygote())
    prob = OptimizationProblem(optprob, x0, _p; sense=GalacticOptim.MaxSense)

    sol = solve(prob, NelderMead())
    @test 10 * sol.minimum < l1

    sol = solve(prob, BFGS())
    @test 10 * sol.minimum < l1

    function g!(G, x)
        G[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
        G[2] = 200.0 * (x[2] - x[1]^2)
    end
    optprob = OptimizationFunction((x, p) -> -rosenbrock(x, p), GalacticOptim.AutoZygote(), grad=g!)
    prob = OptimizationProblem(optprob, x0, _p; sense=GalacticOptim.MaxSense)
    sol = solve(prob, BFGS())
    @test 10 * sol.minimum < l1

    optprob = OptimizationFunction(rosenbrock, GalacticOptim.AutoModelingToolkit())
    prob = OptimizationProblem(optprob, x0, _p)
    sol = solve(prob, Optim.BFGS())
    @test 10 * sol.minimum < l1

    optprob = OptimizationFunction(rosenbrock, GalacticOptim.AutoModelingToolkit(true, false))
    prob = OptimizationProblem(optprob, x0, _p)
    sol = solve(prob, Optim.BFGS())
    @test 10 * sol.minimum < l1
end
