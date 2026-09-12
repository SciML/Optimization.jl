using ChainRulesCore, Enzyme, ForwardDiff, LinearAlgebra, OptimizationBase, Test

function ref_lag_hess!(obj, cons!, x, σ, μ, p)
    function lag(θ)
        res = zeros(eltype(θ), length(μ))
        cons!(res, θ, p)
        return σ * obj(θ, p) + dot(μ, res)
    end
    return ForwardDiff.hessian(lag, x)
end

function ref_lag_hess(obj, cons, x, σ, μ, p)
    lag(θ) = σ * obj(θ, p) + dot(μ, cons(θ, p))
    return ForwardDiff.hessian(lag, x)
end

function check_inplace_clnlbeam(N; σ = 1.0)
    h = 1 / N
    alpha = 350
    x_offset = N + 1
    u_offset = 2(N + 1)
    function objective(x, p)
        return sum(
            0.5 * h * (x[u_offset + i + 1]^2 + x[u_offset + i]^2) +
                0.5 * alpha * h * (cos(x[i + 1]) + cos(x[i])) for i in 1:N
        ) + x[1] * x[2]
    end
    function constraint!(res, x, p)
        for i in 1:N
            res[i] = x[x_offset + i + 1] - x[x_offset + i] -
                0.5 * h * (sin(x[i + 1]) + sin(x[i]))
            res[N + i] = x[i + 1] - x[i] -
                0.5 * h * (x[u_offset + i + 1] + x[u_offset + i])
        end
        return nothing
    end

    # Nonzero point so constraint Hessians (from sin) contribute.
    x = collect(range(0.05; step = 0.01, length = 3(N + 1)))
    multipliers = collect(range(0.25; step = 0.05, length = 2N))
    f = OptimizationFunction(objective, AutoEnzyme(); cons = constraint!)
    instantiated = OptimizationBase.instantiate_function(
        f, x, AutoEnzyme(), nothing, 2N; lag_h = true
    )
    expected = ref_lag_hess!(objective, constraint!, x, σ, multipliers, nothing)

    packed = zeros(length(x) * (length(x) + 1) ÷ 2)
    instantiated.lag_h(packed, x, σ, multipliers)
    @test packed ≈ [expected[i, j] for i in axes(expected, 1) for j in 1:i]

    dense = zeros(length(x), length(x))
    instantiated.lag_h(dense, x, σ, multipliers)
    @test dense ≈ expected
    return nothing
end

function check_inplace_quadratic()
    objective(x, p) = x[1]^2 + x[1] * x[2] + 2 * x[2]^2 + p[1] * x[1] * x[2]
    function constraint!(res, x, p)
        res[1] = x[1]^2 + x[2]^2 - 1
        return nothing
    end

    x = [0.3, -0.4]
    p = [1.5]
    f = OptimizationFunction(objective, AutoEnzyme(); cons = constraint!)
    instantiated = OptimizationBase.instantiate_function(
        f, x, AutoEnzyme(), p, 1; lag_h = true
    )

    @testset "σ = $σ, μ = $μ" for (σ, μ) in (
            (1.0, [1.0]),
            (0.0, [2.5]),
            (2.0, [-1.5]),
            (0.0, [0.0]),
        )
        expected = ref_lag_hess!(objective, constraint!, x, σ, μ, p)
        dense = fill(NaN, 2, 2)
        instantiated.lag_h(dense, x, σ, μ, p)
        @test dense ≈ expected
        packed = fill(NaN, 3)
        instantiated.lag_h(packed, x, σ, μ, p)
        @test packed ≈ [expected[1, 1], expected[2, 1], expected[2, 2]]
        p[1] += 0.25
        x .+= 0.1
    end
    return nothing
end

function check_oop_quadratic(n)
    # Separable quartic objective + quadratic equality; n padded relative to
    # the width-8 Enzyme batch size exercises full and partial batches.
    objective(x, p) = sum(abs2(abs2(xi)) for xi in x) + p[1] * sum(x)
    cons(x, p) = [sum(abs2, x) - 1, x[1] * x[min(2, n)] - p[2]]

    x = collect(range(0.1; step = 0.05, length = n))
    p = [0.3, -0.2]
    μ = [1.25, -0.75]
    f = OptimizationFunction{false}(objective, AutoEnzyme(); cons = cons)
    instantiated = OptimizationBase.instantiate_function(
        f, x, AutoEnzyme(), p, 2; lag_h = true
    )

    for σ in (1.0, 0.0, 2.5)
        expected = ref_lag_hess(objective, cons, x, σ, μ, p)
        H = instantiated.lag_h(x, σ, μ, p)
        @test H isa Matrix
        @test size(H) == (n, n)
        @test H ≈ expected
        @test H ≈ H'
    end
    return nothing
end

@testset "Enzyme Lagrangian Hessian" begin
    enzyme_ext = Base.get_extension(OptimizationBase, :OptimizationEnzymeExt)
    @test enzyme_ext._hessian_batch_width(4) == 4
    @test enzyme_ext._hessian_batch_width(17) == 8

    @testset "in-place clnlbeam N = $N" for N in (1:10..., 20, 40, 60)
        check_inplace_clnlbeam(N)
        check_inplace_clnlbeam(N; σ = 0.0)
    end

    @testset "in-place quadratic σ/μ cases" begin
        check_inplace_quadratic()
    end

    @testset "out-of-place quadratic n = $n" for n in (1, 7, 8, 9, 16, 17)
        check_oop_quadratic(n)
    end
end
