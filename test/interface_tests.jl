using Optimization
using SciMLBase
using Test

struct GenericOptimizationAlgorithm end

mutable struct GenericOptimizationCache <: SciMLBase.AbstractOptimizationCache
    prob::SciMLBase.OptimizationProblem
    alg::GenericOptimizationAlgorithm
    maxiters::Int
end

SciMLBase.has_init(::GenericOptimizationAlgorithm) = true

function SciMLBase.__init(
        prob::SciMLBase.OptimizationProblem,
        alg::GenericOptimizationAlgorithm;
        maxiters = 1,
        kwargs...
    )
    return GenericOptimizationCache(prob, alg, Int(maxiters))
end

function SciMLBase.__solve(cache::GenericOptimizationCache)
    u = copy(cache.prob.u0)
    objective = cache.prob.f(u, cache.prob.p)
    stats = SciMLBase.OptimizationStats(; iterations = cache.maxiters, fevals = 1)
    return SciMLBase.build_solution(
        cache,
        cache.alg,
        u,
        objective;
        retcode = SciMLBase.ReturnCode.Success,
        stats
    )
end

@testset "Generic optimization interface" begin
    f = OptimizationFunction((x, p) -> sum(abs2, x))
    prob = OptimizationProblem(f, [2.0, -1.0])
    alg = GenericOptimizationAlgorithm()

    @test SciMLBase.has_init(alg)
    @test !SciMLBase.allowsbounds(alg)
    @test !SciMLBase.requiresbounds(alg)
    @test !SciMLBase.allowsconstraints(alg)
    @test !SciMLBase.requiresconstraints(alg)

    cache = OptimizationBase.init(prob, alg; maxiters = 3)
    @test cache isa GenericOptimizationCache
    @test cache.maxiters == 3

    cached_solution = OptimizationBase.solve!(cache)
    @test cached_solution isa SciMLBase.AbstractOptimizationSolution
    @test cached_solution.u == prob.u0
    @test cached_solution.objective == 5.0
    @test cached_solution.stats.iterations == 3

    direct_solution = Optimization.solve(prob, alg; maxiters = 2)
    @test direct_solution isa SciMLBase.AbstractOptimizationSolution
    @test direct_solution.objective == 5.0
    @test direct_solution.stats.iterations == 2
end
