using Optimization, OptimizationOptimisers, Random, Test
import ForwardDiff

function adaptive_test_problem()
    mof = MultiObjectiveOptimizationFunction(
        (u, p) -> [(u[1] - 1)^2, 3 * (u[1] + 1)^2], AutoForwardDiff()
    )
    return OptimizationProblem(mof, [0.0])
end

# Detects the silent-noop failure mode: if the callback stops mutating the weights the
# scalarized objective reads, or the shared weight vector is ever replaced/copied, the
# hand-computation tests stay green but this fails.
@testset "solve moves scalarization weights" begin
    prob = adaptive_test_problem()
    sprob = weighted_sum(prob)
    initial = copy(sprob.f.f.weights)
    callback = MiniMax(sprob; every = 1, optimizer = Adam(0.5))
    sol = solve(sprob, Adam(0.05); maxiters = 200, callback, save_best = false)
    updated = callback.weights
    @test updated != initial
    @test all(updated .> initial)
    # The iterate tracks the moving weighted-sum minimizer
    # z*(w) = (w₁ - 3w₂) / (w₁ + 3w₂) of w₁(z-1)² + 3w₂(z+1)².
    w = updated
    @test sol.u[1] ≈ (w[1] - 3 * w[2]) / (w[1] + 3 * w[2]) atol = 1.0e-3
end

# With save_best = true the optimizer re-invokes the callback at the final iteration:
# SoftAdapt's second fire would see unchanged losses, giving a uniform softmax that
# resets the weights to exactly ones. The update guard skips it, so an adapted
# (non-uniform) weight vector survives the solve.
@testset "save_best interaction" begin
    prob = adaptive_test_problem()
    sprob = weighted_sum(prob)
    callback = SoftAdapt(sprob; every = 5)
    solve(sprob, Adam(0.05); maxiters = 10, callback, save_best = true)
    @test callback.weights != [1.0, 1.0]
end
