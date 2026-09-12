using Enzyme, InteractiveUtils, Test

versioninfo()
println("Enzyme ", pkgversion(Enzyme), "; CPU ", Sys.CPU_NAME)
flush(stdout)

function contraction(λ, res)
    value = zero(promote_type(eltype(λ), eltype(res)))
    for i in eachindex(λ, res)
        value += conj(λ[i]) * res[i]
    end
    return value
end

function lagrangian(x, objective, cons, p, λ, σ)
    return σ * objective(x, p) + contraction(λ, cons(x, p))
end

function gradient!(mode, x, dx, f)
    Enzyme.make_zero!(dx)
    Enzyme.autodiff(mode, Const(f), Active, Duplicated(x, dx))
    return nothing
end

function check_case(n, ::Val{W}, σ) where {W}
    objective(x, p) = sum(abs2(abs2(xi)) for xi in x) + p[1] * sum(x)
    cons(x, p) = [sum(abs2, x) - 1, x[1] * x[min(2, n)] - p[2]]
    f = (; f = objective, cons)
    x = collect(range(0.1; step = 0.05, length = n))
    p = [0.3, -0.2]
    μ = [1.25, -0.75]
    lag = x -> lagrangian(x, f.f, f.cons, p, μ, σ)
    dx = zero(x)
    expected_gradient = σ .* (4 .* x .^ 3 .+ p[1]) .+ 2 .* μ[1] .* x
    expected_gradient[1] += μ[2] * x[min(2, n)]
    expected_gradient[min(2, n)] += μ[2] * x[1]
    H = zeros(n, n)
    expected = zeros(n, n)
    for i in 1:n
        expected[i, i] = 12 * σ * x[i]^2 + 2 * μ[1]
    end
    expected[1, min(2, n)] += μ[2]
    expected[min(2, n), 1] += μ[2]
    tangents = ntuple(_ -> zero(x), Val(W))
    for first_index in 1:W:n
        seeds = ntuple(Val(W)) do j
            seed = zero(x)
            first_index + j - 1 <= n && (seed[first_index + j - 1] = 1)
            seed
        end
        Enzyme.make_zero!(dx)
        Enzyme.make_zero!.(tangents)
        Enzyme.autodiff(
            Forward, gradient!, Const(Reverse), BatchDuplicated(x, seeds),
            BatchDuplicated(dx, tangents), Const(lag)
        )
        for j in 1:min(W, n - first_index + 1)
            H[:, first_index + j - 1] .= tangents[j]
        end
    end
    @test H ≈ expected
    println("HESSIAN PASS")
    flush(stdout)
    gradient!(Reverse, x, dx, lag)
    @test dx ≈ expected_gradient
    println("REVERSE PASS")
    flush(stdout)
    return nothing
end

@testset "Standalone Enzyme Hessian" begin
    for width in (1, 7, 8), n in (1, 7, 8, 9, 16, 17), σ in (1.0, 0.0, 2.5)
        println("BEGIN n=", n, " width=", width, " sigma=", σ)
        flush(stdout)
        check_case(n, Val(width), σ)
    end
end
