using OptimizationSpeedMapping, Optimization, ForwardDiff
using Test

@testset "OptimizationSpeedMapping.jl" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 100.0]
    l1 = rosenbrock(x0, _p)
    f = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(f, x0, _p)
    sol = solve(prob, SpeedMappingOpt())
    @test 10 * sol.objective < l1

    prob = OptimizationProblem(f, x0, _p; lb = [-1.0, -1.0], ub = [1.5, 1.5])
    sol = solve(prob, SpeedMappingOpt())
    @test 10 * sol.objective < l1

    f = OptimizationFunction(rosenbrock)
    prob = OptimizationProblem(f, x0, _p)
    sol = solve(prob, SpeedMappingOpt())
    @test 10 * sol.objective < l1

    prob = OptimizationProblem(f, x0, _p; lb = [-1.0, -1.0], ub = [1.5, 1.5])
    sol = solve(prob, SpeedMappingOpt())
    @test 10 * sol.objective < l1

    @testset "cache" begin
        objective(x, p) = (p[1] - x[1])^2
        x0 = zeros(1)
        p = [1.0]

        prob = OptimizationProblem(objective, x0, p)
        cache = Optimization.init(prob, SpeedMappingOpt())
        sol = Optimization.solve!(cache)
        @test sol.u≈[1.0] atol=1e-3

        cache = Optimization.reinit!(cache; p = [2.0])
        sol = Optimization.solve!(cache)
        @test sol.u≈[2.0] atol=1e-3
    end
end
