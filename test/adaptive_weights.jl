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
    # GradientScale with a manual jac produces asymmetric weights, so the iterate
    # tracking z*(w) = (w₁ - 3w₂)/(w₁ + 3w₂) is informative (frozen w = [1,1]
    # would give z* = -0.5, which asymmetric w does not).
    mof = MultiObjectiveOptimizationFunction(
        (u, p) -> [(u[1] - 1)^2, 3 * (u[1] + 1)^2], AutoForwardDiff();
        jac = (J, u, p) -> (J[1, 1] = 2 * (u[1] - 1); J[2, 1] = 6 * (u[1] + 1); J)
    )
    sprob = weighted_sum(OptimizationProblem(mof, [0.0]))
    initial = copy(sprob.f.f.weights)
    callback = GradientScale(sprob; every = 1, inertia = 0.5)
    sol = solve(sprob, Adam(0.05); maxiters = 200, callback, save_best = false)
    updated = callback.weights
    @test updated != initial
    @test updated[1] != updated[2]
    w = updated
    @test sol.u[1] ≈ (w[1] - 3 * w[2]) / (w[1] + 3 * w[2]) atol = 1.0e-3
end

# MiniMax plain ascent also moves the shared weight vector during a solve.
@testset "MiniMax solve moves weights" begin
    prob = adaptive_test_problem()
    sprob = weighted_sum(prob)
    initial = copy(sprob.f.f.weights)
    callback = MiniMax(sprob; every = 1, η = 0.5)
    solve(sprob, Adam(0.05); maxiters = 50, callback, save_best = false)
    @test callback.weights != initial
    @test all(callback.weights .> initial)
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
