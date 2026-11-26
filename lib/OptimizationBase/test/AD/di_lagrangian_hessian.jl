# In-place `lag_h` built through DifferentiationInterface, against ForwardDiff. Enzyme reaches
# this path as `SecondOrder(Enzyme, Enzyme)` (reverse over reverse) and `AutoSparse(Enzyme)`;
# the ForwardDiff backends are controls.
include(joinpath(@__DIR__, "..", "despecialize_common.jl"))
using Enzyme
using ADTypes: ADTypes, AutoSparse
using DifferentiationInterface: SecondOrder
using SparseArrays: nonzeros, sparse

const ENZYME_REVERSE = AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse))

# What `lag_h` fills into a vector: dense backends the lower triangle row by row, sparse ones
# the stored upper-triangle entries column by column. These agree when, as for the Rosenbrock
# problem below, every entry of the Hessian is stored.
packed(H) = [H[i, j] for i in axes(H, 1) for j in 1:i]

function check_rosenbrock(ad)
    of = OptimizationFunction{true}(rosenbrock, ad; cons = rosenbrock_cons!)
    fi = OptimizationBase.instantiate_function(of, copy(X0), ad, P0, 2; lag_h = true)
    for (x, σ, μ) in ((X0, Σ, Μ), (X0, 2.5, -Μ), (X0, Σ, Μ), (zeros(2), Σ, Μ), (X0, 0.0, Μ))
        expected = reference_derivatives(; x, σ, μ).lag_h
        H = fill(NaN, 2, 2)
        fi.lag_h(H, x, σ, μ)
        @test H ≈ expected
        h = fill(NaN, 3)
        fi.lag_h(h, x, σ, μ)
        @test h ≈ packed(expected)
    end

    # The constraints see the `p` passed to `lag_h`, like the objective, also at σ = 0.
    p = RosenbrockParams(2.0, 50.0)
    for σ in (Σ, 0.0)
        expected = reference_derivatives(p; σ).lag_h
        H = fill(NaN, 2, 2)
        fi.lag_h(H, X0, σ, Μ, p)
        @test H ≈ expected
        h = fill(NaN, 3)
        fi.lag_h(h, X0, σ, Μ, p)
        @test h ≈ packed(expected)
    end
    return nothing
end

# A `p` that adds an entry to the sparse Lagrangian Hessian pattern is rejected: its coloring
# and the coordinates of the vector form are those of the pattern at the construction `p`.
function check_pattern_growth(ad)
    obj(x, p) = x[1]^2 + 2x[2]^2
    cons!(res, x, p) = (res[1] = p ? x[1] * x[2] : x[1]^2; nothing)
    of = OptimizationFunction{true}(obj, ad; cons = cons!)
    x = [0.3, 0.7]
    fi = OptimizationBase.instantiate_function(of, copy(x), ad, false, 1; lag_h = true)
    H = similar(fi.lag_hess_prototype, Float64)
    fill!(nonzeros(H), NaN)
    h = fill(NaN, length(first(OptimizationBase.lag_hess_structure(fi.lag_hess_prototype))))
    @test_throws ArgumentError fi.lag_h(H, x, 1.0, [3.0], true)
    @test_throws ArgumentError fi.lag_h(h, x, 1.0, [3.0], true)
    @test all(isnan, nonzeros(H)) && all(isnan, h)
    # a `p` with the same pattern is fine
    fi.lag_h(H, x, 1.0, [3.0], false)
    @test H ≈ [8 0; 0 4]
    # a mutable `p` is checked when a different object is passed
    cons_ref!(res, x, p) = (res[1] = p[] ? x[1] * x[2] : x[1]^2; nothing)
    of_ref = OptimizationFunction{true}(obj, ad; cons = cons_ref!)
    p_ref = Ref(false)
    fi_ref = OptimizationBase.instantiate_function(of_ref, copy(x), ad, p_ref, 1; lag_h = true)
    H_ref = similar(fi_ref.lag_hess_prototype, Float64)
    fi_ref.lag_h(H_ref, x, 1.0, [3.0], p_ref)
    @test H_ref ≈ [8 0; 0 4]
    @test_throws ArgumentError fi_ref.lag_h(H_ref, x, 1.0, [3.0], Ref(true))

    # instantiated at `p = true`, the off-diagonal entry is in the pattern
    fi = OptimizationBase.instantiate_function(of, copy(x), ad, true, 1; lag_h = true)
    S = fi.lag_hess_prototype
    H = similar(S, Float64)
    fi.lag_h(H, x, 1.0, [3.0])
    @test H ≈ [2 3; 3 4]
    h = zeros(length(first(OptimizationBase.lag_hess_structure(S))))
    fi.lag_h(h, x, 1.0, [3.0])
    @test h ≈ [[2 3; 3 4][i, j] for (i, j) in zip(OptimizationBase.lag_hess_structure(S)...)]
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

# At σ = 0 the pattern check leaves out the objective, here undefined at `p = [-1.0]`. A known
# pattern may be dense or have non-`Bool` entries.
function check_pattern_check_inputs()
    obj(x, p) = log(p[1]) * x[1]^2 + x[2]^2
    cons!(res, x, p) = (res[1] = p[1] * x[1]^2; nothing)
    x = [0.3, 0.7]
    @testset "$name" for (name, ad) in (
            ("detected", AutoSparse(AutoForwardDiff())),
            (
                "known, dense Bool", AutoSparse(
                    AutoForwardDiff();
                    sparsity_detector = ADTypes.KnownHessianSparsityDetector(Bool[1 0; 0 1])
                ),
            ),
            (
                "known, sparse Int", AutoSparse(
                    AutoForwardDiff();
                    sparsity_detector = ADTypes.KnownHessianSparsityDetector(sparse([1 0; 0 1]))
                ),
            ),
        )
        of = OptimizationFunction{true}(obj, ad; cons = cons!)
        fi = OptimizationBase.instantiate_function(of, copy(x), ad, [1.0], 1; lag_h = true)
        H = similar(fi.lag_hess_prototype, Float64)
        fi.lag_h(H, x, 0.0, [3.0], [-1.0])
        @test H ≈ [-6 0; 0 0]
        fi.lag_h(H, x, 1.0, [3.0], [2.0])
        @test H ≈ [2log(2) + 12 0; 0 2]
    end
    # a check at σ = 0 leaves out the objective, so σ ≠ 0 with the same `p` checks again
    obj_branch(x, p) = p[] ? x[1] * x[2] : x[1]^2
    ad = AutoSparse(AutoForwardDiff())
    cons_square!(res, x, p) = (res[1] = x[1]^2; nothing)
    of = OptimizationFunction{true}(obj_branch, ad; cons = cons_square!)
    fi = OptimizationBase.instantiate_function(of, copy(x), ad, Ref(false), 1; lag_h = true)
    H = similar(fi.lag_hess_prototype, Float64)
    p_true = Ref(true)
    fi.lag_h(H, x, 0.0, [3.0], p_true)
    @test_throws ArgumentError fi.lag_h(H, x, 1.0, [3.0], p_true)
    return nothing
end

backends = Tuple{String, Any}[
    ("ForwardDiff", AutoForwardDiff()),
    ("AutoSparse(ForwardDiff)", AutoSparse(AutoForwardDiff())),
]
# On Julia 1.10, Enzyme fails to compile these reverse-over-reverse Hessians
# (`EnzymeNoShadowError`, `EnzymeInternalError`).
if VERSION >= v"1.11.0-DEV.0"
    pushfirst!(
        backends,
        ("SecondOrder(Enzyme, Enzyme)", SecondOrder(ENZYME_REVERSE, ENZYME_REVERSE)),
        ("AutoSparse(Enzyme)", AutoSparse(ENZYME_REVERSE)),
    )
end

@testset "in-place Lagrangian Hessian through DifferentiationInterface" begin
    @testset "$name" for (name, ad) in backends
        check_rosenbrock(ad)
        check_clnlbeam(ad, 5)
        ad isa AutoSparse && check_pattern_growth(ad)
    end
    @testset "sparse pattern check inputs" check_pattern_check_inputs()
end
