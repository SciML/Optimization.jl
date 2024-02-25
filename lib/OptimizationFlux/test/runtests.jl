using OptimizationFlux, Optimization, ForwardDiff
using Test

@testset "OptimizationFlux.jl" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 100.0]
    l1 = rosenbrock(x0, _p)

    optprob = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())

    prob = OptimizationProblem(optprob, x0, _p)

    sol = Optimization.solve(prob, Flux.Adam(0.1), maxiters = 1000)
    @test 10 * sol.objective < l1

    prob = OptimizationProblem(optprob, x0, _p)
    sol = solve(prob, Flux.Adam(), maxiters = 1000, progress = false)
    @test 10 * sol.objective < l1

    @testset "cache" begin
        objective(x, p) = (p[1] - x[1])^2
        x0 = zeros(1)
        p = [1.0]

        prob = OptimizationProblem(
            OptimizationFunction(objective,
                Optimization.AutoForwardDiff()), x0,
            p)
        cache = Optimization.init(prob, Flux.Adam(0.1), maxiters = 1000)
        sol = Optimization.solve!(cache)
        @test sol.u≈[1.0] atol=1e-3

        cache = Optimization.reinit!(cache; p = [2.0])
        sol = Optimization.solve!(cache)
        @test sol.u≈[2.0] atol=1e-3
    end

    function cb(state, args...)
        if state.iter % 10 == 0
            println(state.u)
        end
        return false
    end
    sol = solve(prob, Flux.Adam(0.1), callback = cb, maxiters = 100, progress = false)
end
