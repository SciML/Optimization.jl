using OptimizationFlux, Optimization, ForwardDiff
using Test

@testset "OptimizationFlux.jl" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 100.0]
    l1 = rosenbrock(x0, _p)

    optprob = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())

    prob = OptimizationProblem(optprob, x0, _p)

    sol = Optimization.solve(prob, Flux.ADAM(0.1), maxiters = 1000)
    @test 10 * sol.objective < l1

    prob = OptimizationProblem(optprob, x0, _p)
    sol = solve(prob, Flux.ADAM(), maxiters = 1000, progress = false)
    @test 10 * sol.objective < l1
end
