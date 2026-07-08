# Tests for the dual-tolerant gradient/Jacobian path and the `p`-accepting
# derivative closures added on the `dual-tolerant-grad` branch.
#
# The behaviors under test (all previously uncovered):
#   1. `_prep_valid` type-match gating logic in isolation.
#   2. `grad`/`cons_j` still hit the prepared fast path for the construction types
#      and stay numerically correct.
#   3. `grad`/`cons_j` accept an explicit `p` different from the construction `p`.
#   4. Pushing `ForwardDiff.Dual`s (dual `p`, real `θ`) through `grad`/`cons_j`
#      does not throw `PreparationMismatchError` and yields the correct
#      sensitivity (∂/∂p of the derivative) — the SciMLSensitivity use case.
#   5. Any other off-construction eltype (`Float32`, `BigFloat`, …) routes through
#      the prep-free fallback instead of erroring on the prep built at the construction types.

using OptimizationBase, Test, ForwardDiff, FiniteDiff
using ADTypes, Enzyme
import SciMLBase
using OptimizationBase: _prep_valid

# Parametrized objective and constraint whose derivatives genuinely depend on `p`,
# so a dual `p` produces a nonzero, checkable sensitivity.
objp(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
consp!(res, x, p) = (res[1] = p[1] * x[1]^2 + x[2]^2; return nothing)
consp(x, p) = [p[1] * x[1]^2 + x[2]^2]

x0 = zeros(2)
xt = [0.7, -0.3]          # evaluation point, distinct from x0
p0 = [2.0, 3.0]           # construction parameters
p1 = [5.0, 7.0]           # a different parameter value

∇xf(x, p) = ForwardDiff.gradient(xx -> objp(xx, p), x)
consjac(x, p) = ForwardDiff.jacobian(xx -> consp(xx, p), x)

@testset "_prep_valid type-match gating" begin
    d = ForwardDiff.Dual{Nothing}(1.0, 1.0)
    Tx0 = Vector{Float64}   # what the closures capture as the construction type

    # Exact construction type -> prepared fast path.
    @test _prep_valid(Tx0, [1.0, 2.0])
    @test _prep_valid(typeof(SciMLBase.NullParameters()), SciMLBase.NullParameters())
    @test _prep_valid(Nothing, nothing)
    @test _prep_valid(typeof((rand(2, 2), rand(2))), (rand(2, 2), rand(2)))  # structured p

    # Any deviating type -> fallback. Unlike `anyeltypedual`, non-dual types are caught too.
    @test !_prep_valid(Tx0, [d, d])                    # dual
    @test !_prep_valid(Tx0, Float32[1.0, 2.0])         # Float32 (the bug fix)
    @test !_prep_valid(Tx0, big.([1.0, 2.0]))          # BigFloat
    @test !_prep_valid(Tx0, [1, 100])                  # Int
    @test !_prep_valid(typeof((rand(2), [1.0])), ([d, d], [1.0]))  # dual nested in a tuple
end

# Runs the full matrix of assertions against an already-instantiated problem.
# `inplace` selects whether grad/cons_j are the mutating (res, θ[, p]) form.
function check_dual_tolerant(optprob; inplace::Bool, rtol = 1.0e-6)
    # --- gradient ---------------------------------------------------------
    gref = ∇xf(xt, p0)
    if inplace
        g = zeros(2)
        optprob.grad(g, xt)                       # fast path, default p
        @test g ≈ gref rtol = rtol
        optprob.grad(g, xt, p1)                    # explicit different p
        @test g ≈ ∇xf(xt, p1) rtol = rtol
    else
        @test optprob.grad(xt) ≈ gref rtol = rtol
        @test optprob.grad(xt, p1) ≈ ∇xf(xt, p1) rtol = rtol
    end

    # Dual p, real θ: sensitivity ∂/∂p ∇ₓf. Must not throw, must match FD.
    Jsens_ref = ForwardDiff.jacobian(pp -> ∇xf(xt, pp), p0)
    if inplace
        gof_p = pp -> (buf = zeros(eltype(pp), 2); optprob.grad(buf, xt, pp); buf)
    else
        gof_p = pp -> optprob.grad(xt, pp)
    end
    @test ForwardDiff.jacobian(gof_p, p0) ≈ Jsens_ref rtol = rtol

    # --- constraint Jacobian ---------------------------------------------
    Jref = vec(consjac(xt, p0))
    if inplace
        J = zeros(2)
        optprob.cons_j(J, xt)                      # fast path, default p
        @test J ≈ Jref rtol = rtol
        optprob.cons_j(J, xt, p1)                   # explicit different p
        @test J ≈ vec(consjac(xt, p1)) rtol = rtol
    else
        @test optprob.cons_j(xt) ≈ Jref rtol = rtol
        @test optprob.cons_j(xt, p1) ≈ vec(consjac(xt, p1)) rtol = rtol
    end

    # Dual p through cons_j: sensitivity ∂/∂p of the constraint Jacobian.
    Jcons_sens_ref = ForwardDiff.jacobian(pp -> vec(consjac(xt, pp)), p0)
    if inplace
        cjof_p = pp -> (J = zeros(eltype(pp), 2); optprob.cons_j(J, xt, pp); J)
    else
        cjof_p = pp -> optprob.cons_j(xt, pp)
    end
    return @test ForwardDiff.jacobian(cjof_p, p0) ≈ Jcons_sens_ref rtol = rtol
end

@testset "dual-tolerant grad / parametrized cons_j (DI)" begin
    @testset "AutoForwardDiff in-place" begin
        optf = OptimizationFunction(objp, ADTypes.AutoForwardDiff(); cons = consp!)
        optprob = OptimizationBase.instantiate_function(
            optf, x0, ADTypes.AutoForwardDiff(), p0, 1; g = true, cons_j = true
        )
        check_dual_tolerant(optprob; inplace = true)
    end

    @testset "AutoForwardDiff out-of-place" begin
        optf = OptimizationFunction{false}(objp, ADTypes.AutoForwardDiff(); cons = consp)
        optprob = OptimizationBase.instantiate_function(
            optf, x0, ADTypes.AutoForwardDiff(), p0, 1; g = true, cons_j = true
        )
        check_dual_tolerant(optprob; inplace = false)
    end
end

@testset "structured (tuple) parameters" begin
    # Regression: a tuple-valued `p` has a non-`Number` eltype. The output-buffer eltype
    # must not promote to `Union{}` (which crashed the constraint Jacobian), and such a `p`
    # must not veto the prepared fast path — it carries no scalar for the prep to match.
    losst(x, p) = sum(abs2, p[1] * x .- p[2])
    tcons!(res, x, p) = (res[1] = sum(abs2, x) - 1.0; return nothing)
    pt = ([1.0 0.5; 0.5 1.0; 0.2 0.3], [0.1, 0.2, 0.3])   # (Matrix, Vector) tuple

    optf = OptimizationFunction(losst, ADTypes.AutoForwardDiff(); cons = tcons!)
    optprob = OptimizationBase.instantiate_function(
        optf, x0, ADTypes.AutoForwardDiff(), pt, 1; g = true, cons_j = true
    )

    J = zeros(2)
    optprob.cons_j(J, xt)
    @test J ≈ [2xt[1], 2xt[2]] rtol = 1.0e-6         # ∂(‖x‖²-1)/∂x
    g = zeros(2)
    optprob.grad(g, xt)
    @test g ≈ ForwardDiff.gradient(xx -> losst(xx, pt), xt) rtol = 1.0e-6
    # A structured `p`, at its construction type, must still route through the fast path.
    @test _prep_valid(typeof(x0), xt) && _prep_valid(typeof(pt), pt)
end

@testset "foreign θ eltype routes through the fallback (no PreparationMismatchError)" begin
    # A non-dual off-construction eltype (Float32, BigFloat) used to hit the Float64 prep and
    # throw; the type-match gate routes it to the prep-free fallback instead.
    for (inplace, cons) in ((true, consp!), (false, consp))
        optf = inplace ?
            OptimizationFunction(objp, ADTypes.AutoForwardDiff(); cons = cons) :
            OptimizationFunction{false}(objp, ADTypes.AutoForwardDiff(); cons = cons)
        optprob = OptimizationBase.instantiate_function(
            optf, x0, ADTypes.AutoForwardDiff(), p0, 1; g = true, cons_j = true
        )

        for T in (Float32, BigFloat)
            xT = T.(xt)
            gref = ∇xf(xt, p0)         # reference in Float64; compare at loose tol
            if inplace
                gT = zeros(T, 2)
                @test_nowarn optprob.grad(gT, xT)
                @test Float64.(gT) ≈ gref rtol = 1.0e-3
                JT = zeros(T, 2)
                @test_nowarn optprob.cons_j(JT, xT)
                @test Float64.(JT) ≈ vec(consjac(xt, p0)) rtol = 1.0e-3
            else
                @test Float64.(optprob.grad(xT)) ≈ gref rtol = 1.0e-3
                @test Float64.(optprob.cons_j(xT)) ≈ vec(consjac(xt, p0)) rtol = 1.0e-3
            end
        end
    end
end

@testset "parametrized cons_j (Enzyme)" begin
    # The dual-through path is not exercised for Enzyme (ForwardDiff-over-Enzyme
    # nesting is out of scope); this pins the `p`-accepting closure and the
    # de-boxed fast path stay numerically correct at the default and explicit p.
    optf = OptimizationFunction(objp, ADTypes.AutoEnzyme(); cons = consp!)
    optprob = OptimizationBase.instantiate_function(
        optf, x0, ADTypes.AutoEnzyme(), p0, 1; g = true, cons_j = true
    )

    J = zeros(2)
    optprob.cons_j(J, xt)                          # default p
    @test J ≈ vec(consjac(xt, p0)) rtol = 1.0e-6
    optprob.cons_j(J, xt, p1)                       # explicit different p
    @test J ≈ vec(consjac(xt, p1)) rtol = 1.0e-6
end
