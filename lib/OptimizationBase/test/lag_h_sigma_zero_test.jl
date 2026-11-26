using OptimizationBase, Test, DifferentiationInterface
using ADTypes, ForwardDiff, ReverseDiff, Zygote

@testset "Lagrangian Hessian with σ = 0" begin
    # Test that lag_h works correctly when σ = 0
    # This is a regression test for the bug where lag_h! would fail when
    # cons_h was not generated but lag_h needed to compute constraint Hessians

    x0 = [0.5, 0.5]
    rosenbrock(x, p = nothing) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2

    # Single constraint
    cons1 = (res, x, p) -> (res[1] = x[1]^2 + x[2]^2; return nothing)

    # Two constraints
    cons2 = (res, x, p) -> begin
        res[1] = x[1]^2 + x[2]^2
        res[2] = x[2] * sin(x[1]) - x[1]
        return nothing
    end

    @testset "Single constraint with σ = 0" begin
        # Create optimization function WITHOUT cons_h but WITH lag_h
        optf = OptimizationFunction(
            rosenbrock,
            SecondOrder(AutoForwardDiff(), AutoForwardDiff()),
            cons = cons1
        )

        optprob = OptimizationBase.instantiate_function(
            optf, x0,
            SecondOrder(AutoForwardDiff(), AutoForwardDiff()),
            nothing, 1,
            g = true, h = true,
            cons_j = true,
            cons_h = false,  # Don't generate cons_h!
            lag_h = true     # Only generate lag_h!
        )

        # Test with σ = 0 - this should compute only constraint Hessians
        H_lag = Array{Float64}(undef, 2, 2)
        λ = [2.0]  # arbitrary multiplier
        σ = 0.0

        # This should work and compute H = λ[1] * ∇²c₁
        optprob.lag_h(H_lag, x0, σ, λ)

        # Expected: constraint Hessian is [2 0; 0 2] for c(x) = x₁² + x₂²
        # Scaled by λ[1] = 2.0 gives [4 0; 0 4]
        @test H_lag ≈ [4.0 0.0; 0.0 4.0]

        # Test with σ ≠ 0 for comparison
        σ = 1.0
        optprob.lag_h(H_lag, x0, σ, λ)

        # Expected objective Hessian at x0 = [0.5, 0.5]
        H_obj = zeros(2, 2)
        H_obj[1, 1] = 2.0 - 400.0 * x0[2] + 1200.0 * x0[1]^2
        H_obj[1, 2] = -400.0 * x0[1]
        H_obj[2, 1] = -400.0 * x0[1]
        H_obj[2, 2] = 200.0

        # Should be σ * H_obj + λ[1] * H_cons
        @test H_lag ≈ H_obj + [4.0 0.0; 0.0 4.0]
    end

    @testset "Two constraints with σ = 0" begin
        optf = OptimizationFunction(
            rosenbrock,
            SecondOrder(AutoForwardDiff(), AutoForwardDiff()),
            cons = cons2
        )

        optprob = OptimizationBase.instantiate_function(
            optf, x0,
            SecondOrder(AutoForwardDiff(), AutoForwardDiff()),
            nothing, 2,
            g = true, h = true,
            cons_j = true,
            cons_h = false,  # Don't generate cons_h!
            lag_h = true     # Only generate lag_h!
        )

        # Test with σ = 0
        H_lag = Array{Float64}(undef, 2, 2)
        λ = [1.5, -0.5]
        σ = 0.0

        # This should compute H = λ[1] * ∇²c₁ + λ[2] * ∇²c₂
        optprob.lag_h(H_lag, x0, σ, λ)

        # Expected constraint Hessians:
        # ∇²c₁ = [2 0; 0 2] for c₁(x) = x₁² + x₂²
        # ∇²c₂ = [-sin(x₁)*x₂ cos(x₁); cos(x₁) 0] for c₂(x) = x₂*sin(x₁) - x₁
        # At x0 = [0.5, 0.5]:
        H_c2 = zeros(2, 2)
        H_c2[1, 1] = -sin(x0[1]) * x0[2]
        H_c2[1, 2] = cos(x0[1])
        H_c2[2, 1] = cos(x0[1])
        H_c2[2, 2] = 0.0

        expected = λ[1] * [2.0 0.0; 0.0 2.0] + λ[2] * H_c2
        @test H_lag ≈ expected rtol = 1.0e-6
    end

    @testset "Different AD backends with σ = 0" begin
        # Test with AutoReverseDiff
        optf = OptimizationFunction(
            rosenbrock,
            SecondOrder(AutoForwardDiff(), AutoReverseDiff()),
            cons = cons1
        )

        optprob = OptimizationBase.instantiate_function(
            optf, x0,
            SecondOrder(AutoForwardDiff(), AutoReverseDiff()),
            nothing, 1,
            g = true, h = true,
            cons_j = true,
            cons_h = false,
            lag_h = true
        )

        H_lag = Array{Float64}(undef, 2, 2)
        λ = [3.0]
        σ = 0.0

        optprob.lag_h(H_lag, x0, σ, λ)
        @test H_lag ≈ [6.0 0.0; 0.0 6.0]  # 3.0 * [2 0; 0 2]

        # Test with AutoZygote
        optf = OptimizationFunction(
            rosenbrock,
            SecondOrder(AutoForwardDiff(), AutoZygote()),
            cons = cons1
        )

        optprob = OptimizationBase.instantiate_function(
            optf, x0,
            SecondOrder(AutoForwardDiff(), AutoZygote()),
            nothing, 1,
            g = true, h = true,
            cons_j = true,
            cons_h = false,
            lag_h = true
        )

        H_lag = Array{Float64}(undef, 2, 2)
        λ = [0.5]
        σ = 0.0

        optprob.lag_h(H_lag, x0, σ, λ)
        @test H_lag ≈ [1.0 0.0; 0.0 1.0]  # 0.5 * [2 0; 0 2]
    end

    @testset "Edge cases" begin
        optf = OptimizationFunction(
            rosenbrock,
            SecondOrder(AutoForwardDiff(), AutoForwardDiff()),
            cons = cons2
        )

        optprob = OptimizationBase.instantiate_function(
            optf, x0,
            SecondOrder(AutoForwardDiff(), AutoForwardDiff()),
            nothing, 2,
            g = true, h = true,
            cons_j = true,
            cons_h = false,
            lag_h = true
        )

        H_lag = Array{Float64}(undef, 2, 2)

        # Test with all λ = 0 and σ = 0 (should give zero matrix)
        λ = [0.0, 0.0]
        σ = 0.0
        optprob.lag_h(H_lag, x0, σ, λ)
        @test all(H_lag .≈ 0.0)

        # Test with some λ = 0 (should skip those constraints)
        λ = [2.0, 0.0]
        σ = 0.0
        optprob.lag_h(H_lag, x0, σ, λ)
        @test H_lag ≈ [4.0 0.0; 0.0 4.0]  # Only first constraint contributes
    end

    # At σ = 0 the objective is not evaluated (here it is undefined at `x[1] < 0`), and the
    # constraints see the `p` passed to `lag_h`, also with a user-supplied `cons_h`.
    @testset "σ = 0 with a call-time p: $name" for (name, ad, user_cons_h) in [
            ("ForwardDiff", AutoForwardDiff(), false),
            ("ForwardDiff, user cons_h", AutoForwardDiff(), true),
            ("ForwardDiff/ReverseDiff", SecondOrder(AutoForwardDiff(), AutoReverseDiff()), false),
        ]
        log_obj(x, p) = log(x[1]) + x[2]^2
        cons_p!(res, x, p) = (res[1] = p[1] * x[1]^2 + x[2]^2; nothing)
        cons_p_h!(H, x, p) = (H[1] .= [2p[1] 0; 0 2]; nothing)
        optf = if user_cons_h
            OptimizationFunction(log_obj, ad; cons = cons_p!, cons_h = cons_p_h!)
        else
            OptimizationFunction(log_obj, ad; cons = cons_p!)
        end
        optprob = OptimizationBase.instantiate_function(
            optf, x0, ad, [1.0], 1; lag_h = true
        )
        expected = 3.0 * [4.0 0.0; 0.0 2.0] # λ = 3, p = 2
        for x in (x0, [-0.5, 0.5])
            H_lag = fill(NaN, 2, 2)
            optprob.lag_h(H_lag, x, 0.0, [3.0], [2.0])
            @test H_lag ≈ expected
            h = fill(NaN, 3)
            optprob.lag_h(h, x, 0.0, [3.0], [2.0])
            @test h ≈ [expected[i, j] for i in 1:2 for j in 1:i]
        end
    end

    # The out-of-place `lag_h` at σ = 0 returns the weighted sum of the constraint Hessians.
    @testset "out-of-place, σ = 0: $name" for (name, ad) in [
            ("ForwardDiff", AutoForwardDiff()),
            ("AutoSparse(ForwardDiff)", AutoSparse(AutoForwardDiff())),
        ]
        log_obj(x, p) = log(x[1]) + x[2]^2
        cons_p(x, p) = [p[1] * x[1]^2 + x[2]^2, x[1] * x[2]]
        optf = OptimizationFunction{false}(log_obj, ad; cons = cons_p)
        optprob = OptimizationBase.instantiate_function(
            optf, x0, ad, [1.0], 2; lag_h = true
        )
        λ = [3.0, 0.5]
        expected = λ[1] * [4.0 0.0; 0.0 2.0] + λ[2] * [0.0 1.0; 1.0 0.0] # p = 2
        @test Matrix(optprob.lag_h(x0, 0.0, λ, [2.0])) ≈ expected
        @test Matrix(optprob.lag_h([-0.5, 0.5], 0.0, λ, [2.0])) ≈ expected
        @test Matrix(optprob.lag_h(x0, 1.0, λ, [2.0])) ≈ expected + [-4.0 0.0; 0.0 2.0]
    end
end
