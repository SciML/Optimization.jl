using Test
using Optimization
using SciMLBase

@testset "Sense handling" begin
    struct DummyOpt end
    SciMLBase.requiresgradient(::DummyOpt) = true
    SciMLBase.requireshessian(::DummyOpt) = false
    SciMLBase.allowsfg(::DummyOpt) = false
    SciMLBase.allowsfgh(::DummyOpt) = false
    SciMLBase.requiresconsjac(::DummyOpt) = false
    SciMLBase.requiresconshess(::DummyOpt) = false
    SciMLBase.allowsconsvjp(::DummyOpt) = false
    SciMLBase.allowsconsjvp(::DummyOpt) = false
    SciMLBase.requireslagh(::DummyOpt) = false

    obj(x, p) = x[1] + x[2]
    function g!(G, x, p)
        G[1] = 1.0
        G[2] = 1.0
        return G
    end
    x0 = [5.0, 5.0]

    optf = OptimizationFunction(obj; grad = g!)

    prob_min = OptimizationProblem(optf, x0, nothing; sense = Optimization.MinSense)
    cache_min = OptimizationBase.OptimizationCache(prob_min, DummyOpt())
    @test isapprox(cache_min.f.f(x0, nothing), 10.0; atol = 1e-12)
    G_min = similar(x0)
    cache_min.f.grad(G_min, x0)
    @test all(isapprox.(G_min, [1.0, 1.0], atol = 1e-12))

    prob_max = OptimizationProblem(optf, x0, nothing; sense = Optimization.MaxSense)
    cache_max = OptimizationBase.OptimizationCache(prob_max, DummyOpt())
    @test isapprox(cache_max.f.f(x0, nothing), -10.0; atol = 1e-12)
    G_max = similar(x0)
    cache_max.f.grad(G_max, x0)
    @test all(isapprox.(G_max, [-1.0, -1.0], atol = 1e-12))
end
