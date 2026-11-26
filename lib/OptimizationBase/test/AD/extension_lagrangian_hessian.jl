# `lag_h` of the Zygote and Enzyme extensions against the analytic Lagrangian Hessian: the
# constraints see the `p` passed to `lag_h`, and at σ = 0 the objective, undefined at
# `x[1] < 0`, is not evaluated.
using OptimizationBase, Zygote, Enzyme, ChainRulesCore, ForwardDiff, SparseArrays, Test
using ADTypes: ADTypes, AutoForwardDiff, AutoSparse, AutoZygote, AutoEnzyme
using DifferentiationInterface: SecondOrder

objective(x, p) = log(x[1]) + p[2] * x[3]^2
# the same objective as a callable struct
struct Objective end
(::Objective)(x, p) = objective(x, p)
function constraints!(res, x, p)
    res[1] = p[1] * x[1]^2 + x[1] * x[2]
    res[2] = x[2]^3 + x[1] * x[3]
    return nothing
end
constraints(x, p) = [p[1] * x[1]^2 + x[1] * x[2], x[2]^3 + x[1] * x[3]]

# The Hessian has no (2, 3) entry, and its (3, 3) entry comes from the objective only. Its upper
# triangle in CSC order, (1, 1), (1, 2), (2, 2), (1, 3), (3, 3), differs from row order.
function expected_lag_hess(x, σ, λ, p)
    H_obj = [-1 / x[1]^2 0 0; 0 0 0; 0 0 2p[2]]
    H_c1 = [2p[1] 1 0; 1 0 0; 0 0 0]
    H_c2 = [0 0 1; 0 6x[2] 0; 1 0 0]
    return σ * H_obj + λ[1] * H_c1 + λ[2] * H_c2
end

const X0 = [0.5, 0.7, 0.3]
const P0 = [1.0, 1.0]
const LAMBDA = [3.0, -0.5]
const PATTERN = expected_lag_hess(X0, 1.0, [1.0, 1.0], P0) .!= 0

# (θ, σ, call-time p or `nothing` to omit it); at σ = 0 also a θ outside the objective's domain.
const LAG_CASES = [
    (X0, 1.5, nothing),
    (X0, 1.5, [2.0, 3.0]),
    (X0, 0.0, nothing),
    (X0, 0.0, [2.0, 3.0]),
    ([-0.5, 0.7, 0.3], 0.0, nothing),
    ([-0.5, 0.7, 0.3], 0.0, [2.0, 3.0]),
]

call_lag_h!(fi, out, θ, σ, λ, ::Nothing) = fi.lag_h(out, θ, σ, λ)
call_lag_h!(fi, out, θ, σ, λ, p) = fi.lag_h(out, θ, σ, λ, p)

function check_dense(ad, obj = objective)
    of = OptimizationFunction(obj, ad; cons = constraints!)
    fi = OptimizationBase.instantiate_function(of, copy(X0), ad, P0, 2; lag_h = true)
    @testset "θ = $θ, σ = $σ, p = $p" for (θ, σ, p) in LAG_CASES
        expected = expected_lag_hess(θ, σ, LAMBDA, something(p, P0))
        H = fill(NaN, 3, 3)
        call_lag_h!(fi, H, θ, σ, LAMBDA, p)
        @test H ≈ expected
        h = fill(NaN, 6)
        call_lag_h!(fi, h, θ, σ, LAMBDA, p)
        @test h ≈ [expected[i, j] for i in 1:3 for j in 1:i]
    end
    return nothing
end

# The coordinates a solver declares from `lag_hess_prototype`, which a known pattern given as a
# dense matrix makes dense: `lag_hess_structure` of a sparse prototype, the lower triangle row
# by row of a dense one.
function check_sparse(ad; dense_prototype = false)
    of = OptimizationFunction(objective, ad; cons = constraints!)
    fi = OptimizationBase.instantiate_function(of, copy(X0), ad, P0, 2; lag_h = true)
    @test (fi.lag_hess_prototype isa SparseMatrixCSC) == !dense_prototype
    rows, cols = if dense_prototype
        [i for i in 1:3 for j in 1:i], [j for i in 1:3 for j in 1:i]
    else
        OptimizationBase.lag_hess_structure(fi.lag_hess_prototype)
    end
    dense_prototype || @test collect(zip(rows, cols)) == [(1, 1), (1, 2), (2, 2), (1, 3), (3, 3)]
    @testset "θ = $θ, σ = $σ, p = $p" for (θ, σ, p) in LAG_CASES
        expected = expected_lag_hess(θ, σ, LAMBDA, something(p, P0))
        H = similar(fi.lag_hess_prototype, Float64)
        fill!(H isa SparseMatrixCSC ? nonzeros(H) : H, NaN)
        call_lag_h!(fi, H, θ, σ, LAMBDA, p)
        @test Matrix(H) ≈ expected
        h = fill(NaN, length(rows))
        call_lag_h!(fi, h, θ, σ, LAMBDA, p)
        @test h ≈ [expected[i, j] for (i, j) in zip(rows, cols)]
    end
    return nothing
end

# A user-supplied in-place `lag_h` is called with the output, and with the call-time `p`.
function check_user_lag_h(ad)
    user_lag_h!(H, x, σ, λ, p) = (H .= expected_lag_hess(x, σ, λ, p); nothing)
    of = OptimizationFunction(
        objective, ad; cons = constraints!, lag_h = user_lag_h!,
        lag_hess_prototype = sparse(PATTERN)
    )
    fi = OptimizationBase.instantiate_function(of, copy(X0), ad, P0, 2; lag_h = true)
    for p in (nothing, [2.0, 3.0])
        H = fill(NaN, 3, 3)
        call_lag_h!(fi, H, X0, 1.5, LAMBDA, p)
        @test H ≈ expected_lag_hess(X0, 1.5, LAMBDA, something(p, P0))
    end
    return nothing
end

# At σ = 0 with constraints constant in θ, the Lagrangian Hessian is zero.
function check_constant_constraints(ad)
    constant_constraint!(res, x, p) = (res[1] = 1.0; nothing)
    of = OptimizationFunction(objective, ad; cons = constant_constraint!)
    fi = OptimizationBase.instantiate_function(of, copy(X0), ad, P0, 1; lag_h = true)
    sparse_ad = ad isa AutoSparse
    n = sparse_ad ? length(first(OptimizationBase.lag_hess_structure(fi.lag_hess_prototype))) : 6
    for p in (nothing, P0)
        H = sparse_ad ? similar(fi.lag_hess_prototype, Float64) : zeros(3, 3)
        fill!(sparse_ad ? nonzeros(H) : H, NaN)
        call_lag_h!(fi, H, X0, 0.0, [2.0], p)
        @test iszero(H)
        h = fill(NaN, n)
        call_lag_h!(fi, h, X0, 0.0, [2.0], p)
        @test iszero(h)
    end
    return nothing
end

function check_enzyme_oop(obj = objective)
    of = OptimizationFunction{false}(obj, AutoEnzyme(); cons = constraints)
    fi = OptimizationBase.instantiate_function(of, copy(X0), AutoEnzyme(), P0, 2; lag_h = true)
    @testset "θ = $θ, σ = $σ, p = $p" for (θ, σ, p) in LAG_CASES
        H = p === nothing ? fi.lag_h(θ, σ, LAMBDA) : fi.lag_h(θ, σ, LAMBDA, p)
        @test H ≈ expected_lag_hess(θ, σ, LAMBDA, something(p, P0))
    end
    return nothing
end

@testset "Lagrangian Hessian of the Zygote and Enzyme extensions" begin
    @test Base.get_extension(OptimizationBase, :OptimizationZygoteExt) !== nothing
    @test Base.get_extension(OptimizationBase, :OptimizationEnzymeExt) !== nothing

    zygote_fwd = SecondOrder(AutoForwardDiff(), AutoZygote())
    @testset "dense $(nameof(typeof(ad)))" for ad in (AutoZygote(), zygote_fwd, AutoEnzyme())
        check_dense(ad)
    end
    @testset "dense AutoEnzyme, callable struct objective" check_dense(AutoEnzyme(), Objective())
    @testset "AutoSparse($(nameof(typeof(ad))))" for ad in (AutoZygote(), zygote_fwd)
        check_sparse(AutoSparse(ad))
    end
    @testset "AutoSparse(AutoZygote), known dense pattern" begin
        detector = ADTypes.KnownHessianSparsityDetector(PATTERN)
        check_sparse(AutoSparse(AutoZygote(); sparsity_detector = detector); dense_prototype = true)
    end
    @testset "constant constraints, $(nameof(typeof(ad)))" for ad in (
            AutoZygote(), AutoSparse(AutoZygote()), AutoEnzyme(),
        )
        check_constant_constraints(ad)
    end
    @testset "user lag_h, $(nameof(typeof(ad)))" for ad in (
            AutoZygote(), AutoSparse(AutoZygote()), AutoEnzyme(),
        )
        check_user_lag_h(ad)
    end
    @testset "out-of-place Enzyme" begin
        @static if VERSION < v"1.11.0-DEV.0"
            # Enzyme aborts on LTS: https://github.com/EnzymeAD/Enzyme.jl/issues/3574
            @test_broken false
        else
            check_enzyme_oop()
            check_enzyme_oop(Objective())
        end
    end
end
