using Test
using Optimization, Optimization.SciMLBase
using Optimization.ADTypes
using OptimizationODE
using LinearAlgebra

@testset "OptimizationODE Tests" begin

    function f(x, p, args...)
        return sum(abs2, x)
    end

    function g!(g, x, p, args...)
        @. g = 2 * x
    end

    x0 = [2.0, -3.0]
    p = [5.0]

    f_autodiff = OptimizationFunction(f, ADTypes.AutoForwardDiff())
    prob_auto = OptimizationProblem(f_autodiff, x0, p)

    for opt in (ODEGradientDescent(), RKChebyshevDescent(), RKAccelerated(), HighOrderDescent())
        sol = solve(prob_auto, opt; dt=0.01, maxiters=50_000)
        @test sol.u ≈ [0.0, 0.0] atol=1e-2
        @test sol.objective ≈ 0.0 atol=1e-2
        @test sol.retcode == ReturnCode.Success
    end

    f_manual = OptimizationFunction(f, SciMLBase.NoAD(); grad=g!)
    prob_manual = OptimizationProblem(f_manual, x0)

    for opt in (ODEGradientDescent(), RKChebyshevDescent(), RKAccelerated(), HighOrderDescent())
        sol = solve(prob_manual, opt; dt=0.01, maxiters=50_000)
        @test sol.u ≈ [0.0, 0.0] atol=1e-2
        @test sol.objective ≈ 0.0 atol=1e-2
        @test sol.retcode == ReturnCode.Success
    end

    f_fail = OptimizationFunction(f, SciMLBase.NoAD())
    prob_fail = OptimizationProblem(f_fail, x0)

    for opt in (ODEGradientDescent(), RKChebyshevDescent(), RKAccelerated(), HighOrderDescent())
        @test_throws ErrorException solve(prob_fail, opt; dt=0.001, maxiters=20_000)
    end

end
