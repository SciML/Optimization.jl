# In-place `lag_h` built through DifferentiationInterface, against ForwardDiff. Enzyme reaches
# this path as `SecondOrder(Enzyme, Enzyme)` (reverse over reverse) and `AutoSparse(Enzyme)`;
# the ForwardDiff backends are controls.
include(joinpath(@__DIR__, "..", "despecialize_common.jl"))
using Enzyme
using ADTypes: AutoSparse
using DifferentiationInterface: SecondOrder

const ENZYME_REVERSE = AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse))

function sparse_backend(ad)
    return AutoSparse(
        ad; sparsity_detector = OptimizationBase.TracerSparsityDetector(),
        coloring_algorithm = OptimizationBase.GreedyColoringAlgorithm()
    )
end

# What `lag_h` fills into a vector: dense backends the lower triangle row by row, sparse ones
# the stored upper-triangle entries column by column. These agree when, as for the Rosenbrock
# problem below, every entry of the Hessian is stored.
packed(H) = [H[i, j] for i in axes(H, 1) for j in 1:i]

function check_rosenbrock(ad)
    of = OptimizationFunction{true}(rosenbrock, ad; cons = rosenbrock_cons!)
    fi = OptimizationBase.instantiate_function(of, copy(X0), ad, P0, 2; lag_h = true)
    for (σ, μ) in ((Σ, Μ), (2.5, -Μ), (Σ, Μ))
        expected = reference_derivatives(; σ, μ).lag_h
        H = fill(NaN, 2, 2)
        fi.lag_h(H, X0, σ, μ)
        @test H ≈ expected
        h = fill(NaN, 3)
        fi.lag_h(h, X0, σ, μ)
        @test h ≈ packed(expected)
    end

    # The constraints see the `p` passed to `lag_h`, like the objective.
    p = RosenbrockParams(2.0, 50.0)
    expected = reference_derivatives(p).lag_h
    H = fill(NaN, 2, 2)
    fi.lag_h(H, X0, Σ, Μ, p)
    @test H ≈ expected
    return nothing
end

# A larger problem with a sparse Lagrangian Hessian and constraints written as a loop.
function check_clnlbeam(ad, N)
    h = 1 / N
    objective(x, p) = sum(
        0.5 * h * (x[2N + 2 + i + 1]^2 + x[2N + 2 + i]^2) +
            175 * h * (cos(x[i + 1]) + cos(x[i])) for i in 1:N
    )
    function constraints!(res, x, p)
        for i in 1:N
            res[i] = x[N + 1 + i + 1] - x[N + 1 + i] - 0.5 * h * (sin(x[i + 1]) + sin(x[i]))
            res[N + i] = x[i + 1] - x[i] - 0.5 * h * (x[2N + 2 + i + 1] + x[2N + 2 + i])
        end
        return nothing
    end

    x = collect(range(0.05; step = 0.01, length = 3(N + 1)))
    μ = collect(range(0.25; step = 0.05, length = 2N))
    σ = 0.7
    function lagrangian(θ)
        c = zeros(eltype(θ), 2N)
        constraints!(c, θ, nothing)
        return σ * objective(θ, nothing) + dot(μ, c)
    end
    expected = ForwardDiff.hessian(lagrangian, x)

    of = OptimizationFunction{true}(objective, ad; cons = constraints!)
    fi = OptimizationBase.instantiate_function(of, copy(x), ad, nothing, 2N; lag_h = true)
    H = zeros(length(x), length(x))
    fi.lag_h(H, x, σ, μ)
    @test H ≈ expected
    return nothing
end

backends = Tuple{String, Any}[
    ("ForwardDiff", AutoForwardDiff()),
    ("AutoSparse(ForwardDiff)", sparse_backend(AutoForwardDiff())),
]
# On Julia 1.10, Enzyme fails to compile these reverse-over-reverse Hessians
# (`EnzymeNoShadowError`, `EnzymeInternalError`).
if VERSION >= v"1.11.0-DEV.0"
    pushfirst!(
        backends,
        ("SecondOrder(Enzyme, Enzyme)", SecondOrder(ENZYME_REVERSE, ENZYME_REVERSE)),
        ("AutoSparse(Enzyme)", sparse_backend(ENZYME_REVERSE)),
    )
end

@testset "in-place Lagrangian Hessian through DifferentiationInterface" begin
    @testset "$name" for (name, ad) in backends
        check_rosenbrock(ad)
        check_clnlbeam(ad, 5)
    end
end
