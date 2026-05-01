using OptimizationBase
using MLUtils
using OptimizationOptimisers
using OptimizationAuglag
using ForwardDiff
using LinearAlgebra
using Random
using OptimizationBase: OptimizationCache
using SciMLBase: OptimizationFunction, ReturnCode
using Test

# -----------------------------------------------------------------------------
# Fixture: equality-constrained least squares with a closed-form KKT reference.
#
#   min_θ  (1/N) ‖Xᵀθ − y‖²   s.t.  Aθ = b
#
# KKT system  [G Aᵀ; A 0] [θ; μ] = [g; b]  with  G = (2/N) XXᵀ, g = (2/N) Xy
# gives the exact (θ_ref, μ_ref) to compare AugLag's result against.
# -----------------------------------------------------------------------------
function eqls_fixture(; N = 400, d = 8, M = 2, seed = 0x5CA11e,
        batchsize = N, shuffle = false, data_iterator = true)
    rng = Xoshiro(seed)
    θ_true = randn(rng, d)
    X = randn(rng, d, N)
    y = X' * θ_true .+ 0.02 .* randn(rng, N)
    A = randn(rng, M, d)
    b = A * θ_true .+ [0.5, -0.3][1:M]

    G = (2 / N) * X * X'
    g = (2 / N) * X * y
    KKT = [G A'; A zeros(M, M)]
    θ_ref = (KKT \ [g; b])[1:d]

    function loss(θ, batch)
        Xb, yb = batch
        return sum(abs2, Xb' * θ .- yb) / length(yb)
    end
    function cons!(res, θ, _p = nothing)
        res .= A * θ .- b
        return nothing
    end

    p = if data_iterator
        MLUtils.DataLoader((X, y); batchsize = batchsize, shuffle = shuffle)
    else
        (X, y)
    end
    optf = OptimizationFunction(loss, AutoForwardDiff(); cons = cons!)
    prob = OptimizationProblem(optf, zeros(d), p;
        lcons = zeros(M), ucons = zeros(M))

    return (; prob, θ_ref, A, b)
end

@testset "OptimizationAuglag.jl" begin

    @testset "inequality constraint (polynomial fit for sin)" begin
        x0 = (-pi):0.001:pi
        y0 = sin.(x0)
        data = MLUtils.DataLoader((x0, y0), batchsize = 126)

        function loss(coeffs, data)
            ypred = [evalpoly(data[1][i], coeffs) for i in eachindex(data[1])]
            return sum(abs2, ypred .- data[2])
        end
        function cons1(res, coeffs, _p = nothing)
            res[1] = coeffs[1] * coeffs[5] - 1
            return nothing
        end

        optf = OptimizationFunction(loss, AutoSparse(AutoForwardDiff()); cons = cons1)
        initpars = rand(Xoshiro(42), 5)
        l0 = optf(initpars, (x0, y0))
        prob = OptimizationProblem(optf, initpars, data;
            lcons = [-Inf], ucons = [1],
            lb = fill(-10.0, 5), ub = fill(10.0, 5))
        result = solve(prob,
            AugLag(; inner = Adam(), inner_maxiters = 100);
            maxiters = 100)
        @test result.objective < l0
    end

    @testset "equality constraints — Success with KKT agreement" begin
        fx = eqls_fixture()
        result = solve(fx.prob,
            AugLag(; inner = Adam(0.02), inner_maxiters = 200);
            maxiters = 200)

        @test result.retcode === ReturnCode.Success
        @test norm(fx.A * result.u - fx.b, Inf) < 1e-3
        @test norm(result.u - fx.θ_ref) < 0.05
    end

    @testset "ConvergenceFailure when primal unreachable in budget" begin
        # Tight primal tolerance + small budget → primal can't be met.
        fx = eqls_fixture()
        result = solve(fx.prob,
            AugLag(; inner = Adam(0.02), inner_maxiters = 100, ϵ_primal = 1e-12);
            maxiters = 20)
        @test result.retcode === ReturnCode.ConvergenceFailure
    end

    @testset "Terminated when callback returns true" begin
        fx = eqls_fixture()
        fired_at = Ref(0)
        cb = (state, obj) -> begin
            fired_at[] = state.iter
            return state.iter ≥ 3   # stop after 3 outer iters
        end
        result = solve(fx.prob,
            AugLag(; inner = Adam(0.02), inner_maxiters = 50);
            maxiters = 100, callback = cb)
        @test result.retcode === ReturnCode.Terminated
        @test fired_at[] == 3
    end

    @testset "MaxTime enforced at outer loop" begin
        fx = eqls_fixture()
        result = solve(fx.prob,
            AugLag(; inner = Adam(0.02), inner_maxiters = 100);
            maxiters = 10_000, maxtime = 0.01)
        @test result.retcode === ReturnCode.MaxTime
    end

    @testset "no-constraints problem throws" begin
        optf = OptimizationFunction((θ, _p) -> sum(abs2, θ), AutoForwardDiff())
        prob = OptimizationProblem(optf, [1.0, 2.0])
        @test_throws ArgumentError solve(prob,
            AugLag(; inner = Adam(), inner_maxiters = 10); maxiters = 10)
    end

    @testset "stats are populated honestly" begin
        fx = eqls_fixture()
        result = solve(fx.prob,
            AugLag(; inner = Adam(0.02), inner_maxiters = 100);
            maxiters = 50)
        @test result.stats.iterations > 0
        @test result.stats.fevals > 0
        @test result.stats.gevals > 0
        @test result.stats.time > 0
    end

    @testset "ϵ kwarg seeds both ϵ_primal and ϵ_dual (back-compat)" begin
        alg = AugLag(; inner = Adam(), ϵ = 1e-6)
        @test alg.ϵ_primal == 1e-6
        @test alg.ϵ_dual == 1e-6
    end

    @testset "ϵ_primal and ϵ_dual can be set independently" begin
        alg = AugLag(; inner = Adam(), ϵ_primal = 1e-4, ϵ_dual = 1e-2)
        @test alg.ϵ_primal == 1e-4
        @test alg.ϵ_dual == 1e-2
    end

    @testset "defaults for ρmax, progress_window, inner_* kwargs" begin
        alg = AugLag(; inner = Adam())
        @test alg.ρmax == 1.0e12
        @test alg.progress_window == 5
        @test alg.inner_maxiters === nothing
        @test alg.inner_maxtime === nothing
        @test alg.inner_callback === nothing
        @test AugLag(; inner = Adam(), ρmax = 1e6).ρmax == 1e6
        @test AugLag(; inner = Adam(), progress_window = 10).progress_window == 10
    end

    @testset "plain-vector cache.p (non-DataIterator)" begin
        # Same problem as the DataLoader fixture but with `p = (X, y)` passed
        # as a plain tuple. Exercises the non-isa_dataiterator branch of the
        # initial-ρ heuristic and the inner-solve path.
        fx = eqls_fixture(; data_iterator = false)
        result = solve(fx.prob,
            AugLag(; inner = Adam(0.02), inner_maxiters = 200);
            maxiters = 200)

        @test result.retcode === ReturnCode.Success
        @test norm(fx.A * result.u - fx.b, Inf) < 1e-3
        @test norm(result.u - fx.θ_ref) < 0.05
    end

    @testset "ρ_init default is nothing, can be overridden" begin
        @test AugLag(; inner = Adam()).ρ_init === nothing
        @test AugLag(; inner = Adam(), ρ_init = 5.0).ρ_init == 5.0
    end

    @testset "ρ_init override is respected at solve start" begin
        # With γ = 1 the penalty never grows, so the ρ reported in the first
        # outer-iteration callback equals the ρ_init we passed in.
        fx = eqls_fixture(; data_iterator = false)
        seen_ρ = Ref(NaN)
        cb = (state, _obj) -> begin
            if state.iter == 1
                seen_ρ[] = state.original.ρ
            end
            return state.iter ≥ 1   # stop after one outer iter
        end
        ρ_init = 42.0
        solve(fx.prob,
            AugLag(; inner = Adam(0.02), inner_maxiters = 10,
                γ = 1.0, ρ_init = ρ_init);
            maxiters = 5, callback = cb)
        @test seen_ρ[] == ρ_init
    end

    @testset "lower-bounded inequality (c ≥ lcons)" begin
        # min ‖θ - target‖²  s.t.  θ₁ + θ₂ ≥ 1.
        # Unconstrained minimizer is `target = (0.0, 0.0)`, infeasible.
        # Active-constraint optimum: project target onto θ₁ + θ₂ = 1, giving
        # (0.5, 0.5).
        target = [0.0, 0.0]
        loss(θ, _p) = sum(abs2, θ .- target)
        cons!(res, θ, _p = nothing) = (res[1] = θ[1] + θ[2]; nothing)

        optf = OptimizationFunction(loss, AutoForwardDiff(); cons = cons!)
        prob = OptimizationProblem(optf, [-1.0, -1.0], nothing;
            lcons = [1.0], ucons = [Inf])
        result = solve(prob,
            AugLag(; inner = Adam(0.05), inner_maxiters = 200);
            maxiters = 200)

        @test result.retcode === ReturnCode.Success
        @test result.u[1] + result.u[2] ≥ 1 - 1.0e-3
        @test norm(result.u .- [0.5, 0.5], Inf) < 5.0e-2
    end

    @testset "two-sided inequality (lcons ≤ c ≤ ucons)" begin
        # min (θ₁ - 2)² + (θ₂ + 1)²  s.t.  −0.5 ≤ θ₁ + θ₂ ≤ 0.5.
        # Unconstrained minimizer (2, −1) has θ₁+θ₂ = 1 > 0.5 → upper-active.
        # Project onto θ₁ + θ₂ = 0.5: optimum is (1.75, −1.25).
        target = [2.0, -1.0]
        loss(θ, _p) = (θ[1] - target[1])^2 + (θ[2] - target[2])^2
        cons!(res, θ, _p = nothing) = (res[1] = θ[1] + θ[2]; nothing)

        optf = OptimizationFunction(loss, AutoForwardDiff(); cons = cons!)
        prob = OptimizationProblem(optf, [0.0, 0.0], nothing;
            lcons = [-0.5], ucons = [0.5])
        result = solve(prob,
            AugLag(; inner = Adam(0.05), inner_maxiters = 200);
            maxiters = 200)

        @test result.retcode === ReturnCode.Success
        @test -0.5 - 1.0e-3 ≤ result.u[1] + result.u[2] ≤ 0.5 + 1.0e-3
        @test norm(result.u .- [1.75, -1.25], Inf) < 5.0e-2
    end

    @testset "two-sided inequality activates lower side" begin
        # Same constraint −0.5 ≤ θ₁ + θ₂ ≤ 0.5, but with a target that pulls
        # the sum below the lower bound: target (−2, 0) → unconstrained sum −2.
        # Project onto θ₁ + θ₂ = −0.5: optimum is (−1.25, 0.75).
        target = [-2.0, 0.0]
        loss(θ, _p) = (θ[1] - target[1])^2 + (θ[2] - target[2])^2
        cons!(res, θ, _p = nothing) = (res[1] = θ[1] + θ[2]; nothing)

        optf = OptimizationFunction(loss, AutoForwardDiff(); cons = cons!)
        prob = OptimizationProblem(optf, [0.0, 0.0], nothing;
            lcons = [-0.5], ucons = [0.5])
        result = solve(prob,
            AugLag(; inner = Adam(0.05), inner_maxiters = 200);
            maxiters = 200)

        @test result.retcode === ReturnCode.Success
        @test -0.5 - 1.0e-3 ≤ result.u[1] + result.u[2] ≤ 0.5 + 1.0e-3
        @test norm(result.u .- [-1.25, 0.75], Inf) < 5.0e-2
    end

    @testset "cons! sees cache.p (DataLoader), not per-batch slices" begin
        # AugLag treats constraints as non-stochastic / batch-independent:
        # `cons!(res, θ, p)` must always be called with the full `cache.p`
        # (here the DataLoader iterator itself), never with a per-batch
        # tuple, even though the inner Adam solve iterates over batches.
        # A regression that routed per-batch `p` into `cons!` would show
        # up as cons-tuple entries in `seen_ps` instead of the iterator.
        rng = Xoshiro(0x5CA11e)
        N, d, M = 80, 3, 1
        X = randn(rng, d, N)
        y = X' * randn(rng, d) .+ 0.01 .* randn(rng, N)
        A = randn(rng, M, d)
        dl = MLUtils.DataLoader((X, y); batchsize = 20, shuffle = false)

        seen_ps = Any[]
        loss(θ, batch) = sum(abs2, batch[1]' * θ .- batch[2]) / length(batch[2])
        function cons_record!(res, θ, p)
            push!(seen_ps, p)
            res .= A * θ
            return nothing
        end

        optf = OptimizationFunction(loss, AutoForwardDiff(); cons = cons_record!)
        prob = OptimizationProblem(optf, zeros(d), dl;
            lcons = zeros(M), ucons = zeros(M))

        solve(prob,
            AugLag(; inner = Adam(0.02), inner_maxiters = 4);
            maxiters = 2)

        @test !isempty(seen_ps)
        n_dl = count(p -> p === dl, seen_ps)
        n_other = count(p -> p !== dl, seen_ps)
        # The DI cons_j preparation calls cons once or twice with the
        # closed-over first batch — those are the only legal non-DL
        # entries. Every call from `__solve` onward must be the iterator.
        @test n_dl ≥ 1
        @test n_dl > n_other
    end

    @testset "degenerate pure-penalty (γ=1, λ=μ=0)" begin
        # Pure quadratic penalty: γ = 1 freezes ρ at ρ_init, and clamping the
        # multiplier bounds to zero forces λ ≡ 0, μ ≡ 0. Only the penalty term
        # drives feasibility, so we need ρ_init large enough to meet ϵ_primal.
        fx = eqls_fixture(; data_iterator = false)
        ρ_init = 1.0e4
        ϵ_primal = 1.0e-2
        result = solve(fx.prob,
            AugLag(; inner = Adam(0.05), inner_maxiters = 500,
                γ = 1.0, ρ_init = ρ_init,
                λmin = 0.0, λmax = 0.0,
                μmin = 0.0, μmax = 0.0,
                ϵ_primal = ϵ_primal, ϵ_dual = ϵ_primal * ρ_init);
            maxiters = 50)

        @test result.retcode === ReturnCode.Success
        @test norm(fx.A * result.u - fx.b, Inf) < ϵ_primal
    end
end
