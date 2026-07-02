# Tests for the dual-tolerant gradient/Jacobian path and the `p`-accepting
# derivative closures added on the `dual-tolerant-grad` branch.
#
# The behaviors under test (all previously uncovered):
#   1. `_use_prep` dual-detection gating logic in isolation.
#   2. `grad`/`cons_j` still hit the prepared fast path for the construction types
#      and stay numerically correct.
#   3. `grad`/`cons_j` accept an explicit `p` different from the construction `p`.
#   4. Pushing `ForwardDiff.Dual`s (dual `p`, real `θ`) through `grad`/`cons_j`
#      does not throw `PreparationMismatchError` and yields the correct
#      sensitivity (∂/∂p of the derivative) — the SciMLSensitivity use case.

using OptimizationBase, Test, ForwardDiff, FiniteDiff
using ADTypes, Enzyme
import SciMLBase
using OptimizationBase: _use_prep

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

@testset "_use_prep dual gating" begin
    d = ForwardDiff.Dual{Nothing}(1.0, 1.0)
    # No duals -> prepared fast path, regardless of real parameter eltype.
    @test _use_prep([1.0, 2.0], [3.0, 4.0])
    @test _use_prep([1.0, 2.0], SciMLBase.NullParameters())
    @test _use_prep([1.0, 2.0], nothing)
    @test _use_prep([1.0, 2.0], [1, 100])            # Int p keeps the fast path
    @test _use_prep([1.0, 2.0], Float32[1, 2])       # mixed-precision p keeps it too
    @test _use_prep([1.0, 2.0], (rand(2, 2), rand(2)))  # structured (tuple) p keeps it
    @test _use_prep(Float32[1.0, 2.0], [3.0, 4.0])   # off-construction real eltype: still fast
    # Dual θ -> fallback.
    @test !_use_prep([d, d], [3.0, 4.0])
    # Real θ but dual p (the sensitivity case) -> fallback.
    @test !_use_prep([1.0, 2.0], [d, d])
    # Dual nested inside a structured p -> fallback (anyeltypedual recurses).
    @test !_use_prep([1.0, 2.0], ([d, d], [1.0]))
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
    # A structured `p` must still route through the prepared fast path.
    @test _use_prep(xt, pt)
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
