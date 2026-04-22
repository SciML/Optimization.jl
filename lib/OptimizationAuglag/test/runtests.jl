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
        batchsize = N, shuffle = false)
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

    data = MLUtils.DataLoader((X, y); batchsize = batchsize, shuffle = shuffle)
    optf = OptimizationFunction(loss, AutoForwardDiff(); cons = cons!)
    prob = OptimizationProblem(optf, zeros(d), data;
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
        result = solve(prob, AugLag(; inner = Adam()); maxiters = 10000)
        @test result.objective < l0
    end

    @testset "equality constraints — Success with KKT agreement" begin
        fx = eqls_fixture()
        result = solve(fx.prob, AugLag(; inner = Adam(0.02)); maxiters = 2000)

        @test result.retcode === ReturnCode.Success
        @test norm(fx.A * result.u - fx.b, Inf) < 1e-3
        @test norm(result.u - fx.θ_ref) < 0.05
    end

    @testset "ConvergenceFailure when primal unreachable in budget" begin
        # Tight primal tolerance + tiny budget → primal can't be met.
        fx = eqls_fixture()
        result = solve(fx.prob,
            AugLag(; inner = Adam(0.02), ϵ_primal = 1e-12);
            maxiters = 50)
        @test result.retcode === ReturnCode.ConvergenceFailure
    end

    @testset "stats are populated honestly" begin
        fx = eqls_fixture()
        result = solve(fx.prob, AugLag(; inner = Adam(0.02)); maxiters = 500)
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

    @testset "ρmax and progress_window have sensible defaults" begin
        alg = AugLag(; inner = Adam())
        @test alg.ρmax == 1.0e12
        @test alg.progress_window == 5
        @test AugLag(; inner = Adam(), ρmax = 1e6).ρmax == 1e6
        @test AugLag(; inner = Adam(), progress_window = 10).progress_window == 10
    end
end
