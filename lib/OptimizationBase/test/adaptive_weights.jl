using OptimizationBase, Random, Test
import ForwardDiff

function adaptive_test_problem(; adtype = NoAD())
    mof = MultiObjectiveOptimizationFunction(
        (u, p) -> [(u[1] - 1)^2, 3 * (u[1] + 1)^2], adtype;
        jac = (J, u, p) -> (J[1, 1] = 2 * (u[1] - 1); J[2, 1] = 6 * (u[1] + 1); J)
    )
    return weighted_sum(OptimizationProblem(mof, [0.0]))
end

adaptive_state(prob, iter, z) = OptimizationBase.OptimizationState(
    iter = iter, u = [z], p = prob.p
)

@testset "Adaptive weight rules match hand computations" begin
    @testset "GradientScale (manual jac, scalar u)" begin
        prob = adaptive_test_problem()
        callback = GradientScale(prob; inertia = 0.0)
        callback(adaptive_state(prob, 1, 0.0), 0.0)
        # ∇L = (-2, 6) at z = 0: proposed = [6/2, 6/6]
        @test callback.weights ≈ [3.0, 1.0]
    end

    @testset "GradientScale (adtype jacobian)" begin
        prob = adaptive_test_problem(; adtype = AutoForwardDiff())
        callback = GradientScale(prob; inertia = 0.0)
        callback(adaptive_state(prob, 1, 0.0), 0.0)
        @test callback.weights ≈ [3.0, 1.0]
    end

    # 2-D u where mean|∇Lᵢ| ≠ max|∇Lᵢ|, so a max-for-mean mutant fails.
    # L₁ = u₁² + 10 u₂², L₂ = (u₁ - 1)² + u₂² at (1, 1):
    # ∇L₁ = (2, 20), ∇L₂ = (0, 2); scale = 20; means = 11, 1 → [20/11, 20].
    @testset "GradientScale (2-D, mean ≠ max)" begin
        mof = MultiObjectiveOptimizationFunction(
            (u, p) -> [u[1]^2 + 10 * u[2]^2, (u[1] - 1)^2 + u[2]^2],
            AutoForwardDiff()
        )
        prob = weighted_sum(OptimizationProblem(mof, [1.0, 1.0]))
        callback = GradientScale(prob; inertia = 0.0)
        callback(
            OptimizationBase.OptimizationState(iter = 1, u = [1.0, 1.0], p = prob.p),
            0.0
        )
        @test callback.weights ≈ [20 / 11, 20]
    end

    @testset "MiniMax" begin
        # Plain ascent w ← w + η L: at z = 0, L = (1, 3), η = 0.5 → [1.5, 2.5]
        prob = adaptive_test_problem()
        callback = MiniMax(prob; η = 0.5)
        callback(adaptive_state(prob, 1, 0.0), 0.0)
        @test callback.weights ≈ [1.5, 2.5]
    end

    @testset "SoftAdapt" begin
        prob = adaptive_test_problem()
        callback = SoftAdapt(prob; α = 0.1)
        callback(adaptive_state(prob, 1, 0.0), 0.0)
        callback(adaptive_state(prob, 2, 0.5), 0.0)
        scores = 0.1 .* [-0.75, 1.25]
        expected = 2 .*
            exp.(scores .- maximum(scores)) ./
            sum(exp.(scores .- maximum(scores)))
        @test callback.weights ≈ expected
    end

    @testset "ReLoBRaLo (α ≠ 0.5, β = 0)" begin
        # α = 0.99 so swapping α with (1-α) changes the result.
        prob = adaptive_test_problem()
        callback = ReLoBRaLo(
            prob; α = 0.99, β = 0.0, temperature = 0.7, rng = Xoshiro(2)
        )
        callback(adaptive_state(prob, 1, 0.0), 0.0)
        callback(adaptive_state(prob, 2, 0.25), 0.0)
        callback(adaptive_state(prob, 3, 0.5), 0.0)
        initial = [1.0, 3.0]
        previous = [0.5625, 4.6875]
        current = [0.25, 6.75]
        scores0 = current ./ (0.7 .* initial)
        scores1 = current ./ (0.7 .* previous)
        initial_balance = 2 .* exp.(scores0 .- maximum(scores0)) ./
            sum(exp.(scores0 .- maximum(scores0)))
        previous_balance = 2 .* exp.(scores1 .- maximum(scores1)) ./
            sum(exp.(scores1 .- maximum(scores1)))
        # β = 0 selects the initial-loss balance deterministically
        expected = 0.99 .* initial_balance .+ 0.01 .* previous_balance
        @test callback.weights ≈ expected
    end

    @testset "ReLoBRaLo (β = 1 carries previous weights)" begin
        prob = adaptive_test_problem()
        callback = ReLoBRaLo(
            prob; α = 0.99, β = 1.0, temperature = 0.7, rng = Xoshiro(2)
        )
        callback(adaptive_state(prob, 1, 0.0), 0.0)
        callback(adaptive_state(prob, 2, 0.25), 0.0)
        previous_weights = copy(callback.weights)
        callback(adaptive_state(prob, 3, 0.5), 0.0)
        previous = [0.5625, 4.6875]
        current = [0.25, 6.75]
        scores1 = current ./ (0.7 .* previous)
        previous_balance = 2 .* exp.(scores1 .- maximum(scores1)) ./
            sum(exp.(scores1 .- maximum(scores1)))
        # β = 1 always takes rho = true → carry previous_weights
        expected = 0.99 .* previous_weights .+ 0.01 .* previous_balance
        @test callback.weights ≈ expected
    end
end

@testset "Update guard" begin
    prob = adaptive_test_problem()
    callback = MiniMax(prob; η = 0.5)
    callback(adaptive_state(prob, 1, 0.0), 0.0)
    once = copy(callback.weights)
    # A second call at the same iteration (the save_best finalization path) is skipped.
    callback(adaptive_state(prob, 1, 0.0), 0.0)
    @test callback.weights == once
    callback(adaptive_state(prob, 2, 0.0), 0.0)
    @test callback.weights != once
end

@testset "Update guard resets on iteration rewind" begin
    prob = adaptive_test_problem()
    callback = MiniMax(prob; every = 5, η = 0.5)
    callback(adaptive_state(prob, 5, 0.0), 0.0)
    after_first_solve = copy(callback.weights)
    # Simulated second solve reusing the same callback: iter goes backwards past the
    # last update, so the guard must not drop the new solve's iter-5 update.
    callback(adaptive_state(prob, 1, 0.0), 0.0)
    callback(adaptive_state(prob, 5, 0.0), 0.0)
    @test callback.weights != after_first_solve
end

@testset "weighted_sum scalarization" begin
    prob = adaptive_test_problem()
    @test prob.f.f isa OptimizationBase.WeightedSumObjective
    @test prob.f.f([0.0], prob.p) == 4.0
    # grad is J'w under the current weights: J = [[-2], [6]] at z = 0, w = [1, 1]
    G = zeros(1)
    prob.f.grad(G, [0.0], prob.p)
    @test G == [4.0]

    mof = MultiObjectiveOptimizationFunction((u, p) -> [(u[1] - 1)^2, 3 * (u[1] + 1)^2])
    @test_throws ArgumentError weighted_sum(
        OptimizationProblem(OptimizationFunction((u, p) -> u[1]^2), [0.0])
    )
    @test_throws DimensionMismatch weighted_sum(
        OptimizationProblem(mof, [0.0]); weights = [1.0, 1.0, 1.0]
    )
    @test_throws ArgumentError MiniMax(prob; every = 0)
    @test_throws ArgumentError GradientScale(
        weighted_sum(OptimizationProblem(mof, [0.0]))
    )
end
