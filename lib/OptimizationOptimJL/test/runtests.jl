using OptimizationOptimJL,
    OptimizationOptimJL.Optim, Optimization, ForwardDiff, Zygote,
    Random, ModelingToolkit
using Test

@testset "OptimizationOptimJL.jl" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 100.0]
    l1 = rosenbrock(x0, _p)

    prob = OptimizationProblem(rosenbrock, x0, _p)
    sol = solve(prob,
        Optim.NelderMead(;
            initial_simplex = Optim.AffineSimplexer(; a = 0.025,
                b = 0.5)))
    @test 10 * sol.objective < l1

    f = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())

    Random.seed!(1234)
    prob = OptimizationProblem(f, x0, _p, lb = [-1.0, -1.0], ub = [0.8, 0.8])
    sol = solve(prob, SAMIN())
    @test 10 * sol.objective < l1

    prob = OptimizationProblem(f, x0, _p)
    Random.seed!(1234)
    sol = solve(prob, SimulatedAnnealing())
    @test 10 * sol.objective < l1

    sol = solve(prob, Optim.BFGS())
    @test 10 * sol.objective < l1

    sol = solve(prob, Optim.Newton())
    @test 10 * sol.objective < l1

    sol = solve(prob, Optim.KrylovTrustRegion())
    @test 10 * sol.objective < l1

    sol = solve(prob, Optim.BFGS(), maxiters = 1)
    @test sol.original.iterations == 1

    sol = solve(prob, Optim.BFGS(), maxiters = 1, local_maxiters = 2)
    @test sol.original.iterations == 1

    sol = solve(prob, Optim.BFGS(), local_maxiters = 2)
    @test sol.original.iterations > 2

    cons = (res, x, p) -> res .= [x[1]^2 + x[2]^2]
    optprob = OptimizationFunction(rosenbrock, Optimization.AutoModelingToolkit();
        cons = cons)

    prob = OptimizationProblem(optprob, x0, _p, lcons = [-5.0], ucons = [10.0])
    sol = solve(prob, IPNewton())
    @test 10 * sol.objective < l1

    optprob = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff();
        cons = cons)

    prob = OptimizationProblem(optprob, x0, _p, lcons = [-Inf], ucons = [Inf])
    sol = solve(prob, IPNewton())
    @test 10 * sol.objective < l1

    prob = OptimizationProblem(optprob, x0, _p, lcons = [-Inf], ucons = [Inf],
        lb = [-500.0, -500.0], ub = [50.0, 50.0])
    sol = solve(prob, IPNewton())
    @test sol.objective < l1

    function con2_c(res, x, p)
        res .= [x[1]^2 + x[2]^2, x[2] * sin(x[1]) - x[1]]
    end

    optprob = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff();
        cons = con2_c)
    prob = OptimizationProblem(optprob, x0, _p, lcons = [-Inf, -Inf], ucons = [Inf, Inf])
    sol = solve(prob, IPNewton())
    @test 10 * sol.objective < l1

    cons_circ = (res, x, p) -> res .= [x[1]^2 + x[2]^2]
    optprob = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff();
        cons = cons_circ)
    prob = OptimizationProblem(optprob, x0, _p, lcons = [-Inf], ucons = [0.25^2])
    cache = Optimization.init(prob, Optim.IPNewton())
    sol = Optimization.solve!(cache)
    res = Array{Float64}(undef, 1)
    cons(res, sol.u, nothing)
    @test sqrt(res[1])≈0.25 rtol=1e-6

    optprob = OptimizationFunction(rosenbrock, Optimization.AutoZygote())

    prob = OptimizationProblem(optprob, x0, _p, lb = [-1.0, -1.0], ub = [0.8, 0.8])
    sol = solve(prob, Optim.Fminbox())
    @test 10 * sol.objective < l1

    Random.seed!(1234)
    prob = OptimizationProblem(optprob, x0, _p, lb = [-1.0, -1.0], ub = [0.8, 0.8])
    cache = Optimization.init(prob, Optim.SAMIN())
    sol = Optimization.solve!(cache)
    @test 10 * sol.objective < l1

    optprob = OptimizationFunction((x, p) -> -rosenbrock(x, p), Optimization.AutoZygote())
    prob = OptimizationProblem(optprob, x0, _p; sense = Optimization.MaxSense)

    sol = solve(prob, NelderMead())
    @test 10 * sol.objective < l1

    sol = solve(prob, BFGS())
    @test 10 * sol.objective < l1

    function g!(G, x, p = nothing)
        G[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
        G[2] = 200.0 * (x[2] - x[1]^2)
    end
    optprob = OptimizationFunction((x, p) -> -rosenbrock(x, p), Optimization.AutoZygote(),
        grad = g!)
    prob = OptimizationProblem(optprob, x0, _p; sense = Optimization.MaxSense)
    sol = solve(prob, BFGS())
    @test 10 * sol.objective < l1

    optprob = OptimizationFunction(rosenbrock, Optimization.AutoModelingToolkit())
    prob = OptimizationProblem(optprob, x0, _p)
    sol = solve(prob, Optim.BFGS())
    @test 10 * sol.objective < l1

    optprob = OptimizationFunction(rosenbrock,
        Optimization.AutoModelingToolkit(true, false))
    prob = OptimizationProblem(optprob, x0, _p)
    sol = solve(prob, Optim.Newton())
    @test 10 * sol.objective < l1

    sol = solve(prob, Optim.KrylovTrustRegion())
    @test 10 * sol.objective < l1

    @testset "cache" begin
        objective(x, p) = (p[1] - x[1])^2
        x0 = zeros(1)
        p = [1.0]

        prob = OptimizationProblem(objective, x0, p)
        cache = Optimization.init(prob, Optim.NelderMead())
        sol = Optimization.solve!(cache)
        @test sol.u≈[1.0] atol=1e-3

        cache = Optimization.reinit!(cache; p = [2.0])
        sol = Optimization.solve!(cache)
        @test sol.u≈[2.0] atol=1e-3
    end
end
