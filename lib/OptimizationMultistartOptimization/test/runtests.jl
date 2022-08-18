using Pkg;
Pkg.develop(path = joinpath(@__DIR__, "../../", "OptimizationNLopt"));
using OptimizationMultistartOptimization, Optimization, ForwardDiff, OptimizationNLopt
using Test

@testset "OptimizationMultistartOptimization.jl" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 100.0]
    l1 = rosenbrock(x0, _p)
    f = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
    prob = Optimization.OptimizationProblem(f, x0, _p, lb = [-1.0, -1.0], ub = [1.5, 1.5])
    sol = solve(prob, OptimizationMultistartOptimization.TikTak(100),
                OptimizationNLopt.Opt(:LD_LBFGS, 2))
    @test 10 * sol.minimum < l1
end
