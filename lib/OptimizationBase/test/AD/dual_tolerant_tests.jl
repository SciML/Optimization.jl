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
using ADTypes, Enzyme, StaticArrays
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

# Second-order / combined DI preps (hess, hv, fgh, fg, lag_h) gate on construction
# types and fall back to prep-free DI when θ/p/v types differ. Each call must match
# a fresh-prep result. Matching-type fgh uses value_gradient_and_hessian(!).
Hess_ref(x, p) = ForwardDiff.hessian(xx -> objp(xx, p), x)
Hv_ref(x, p, v) = Hess_ref(x, p) * v

@testset "matching-type fgh! (IIP/OOP)" begin
    ad = ADTypes.AutoForwardDiff()
    fi = OptimizationBase.instantiate_function(
        OptimizationFunction(objp, ad), x0, ad, p0, 0; fgh = true
    )
    G = zeros(2)
    H = zeros(2, 2)
    y = fi.fgh(G, H, xt)
    @test y ≈ objp(xt, p0) rtol = 1.0e-6
    @test G ≈ ∇xf(xt, p0) rtol = 1.0e-6
    @test H ≈ Hess_ref(xt, p0) rtol = 1.0e-6

    oop = OptimizationBase.instantiate_function(
        OptimizationFunction{false}(objp, ad), x0, ad, p0, 0; fgh = true
    )
    yo, Go, Ho = oop.fgh(xt)
    @test yo ≈ objp(xt, p0) rtol = 1.0e-6
    @test Go ≈ ∇xf(xt, p0) rtol = 1.0e-6
    @test Ho ≈ Hess_ref(xt, p0) rtol = 1.0e-6
end

@testset "foreign types through hess!/hv!/fg!/fgh!/lag_h! (DI prep fallback)" begin
    ad = ADTypes.AutoForwardDiff()
    optf = OptimizationFunction(objp, ad; cons = consp!)
    fi = OptimizationBase.instantiate_function(
        optf, x0, ad, p0, 1; g = true, h = true, hv = true, fg = true, fgh = true, lag_h = true
    )
    oop = OptimizationBase.instantiate_function(
        OptimizationFunction{false}(objp, ad), x0, ad, p0, 0;
        g = true, h = true, hv = true, fg = true, fgh = true
    )

    # --- hess!(H, x::Float32) ---
    H32 = zeros(Float32, 2, 2)
    fi.hess(H32, Float32.(xt))
    @test Float64.(H32) ≈ Hess_ref(xt, p0) rtol = 1.0e-3
    fresh_h = OptimizationBase.instantiate_function(
        optf, Float32.(x0), ad, Float32.(p0), 1; h = true
    )
    H32b = zeros(Float32, 2, 2)
    fresh_h.hess(H32b, Float32.(xt))
    @test H32 ≈ H32b

    # Dual `p` through second-order AutoForwardDiff is a ForwardDiff tag-nesting
    # limitation (fresh prep at Dual `p` throws the same DualMismatchError), so it
    # is not asserted here. See first-order Dual `p` above.

    # --- hv!(Hv, x::BigFloat, v) ---
    xB = big.(xt)
    vB = BigFloat[1, 0]
    HvB = zeros(BigFloat, 2)
    fi.hv(HvB, xB, vB)
    @test Float64.(HvB) ≈ Hv_ref(xt, p0, Float64.(vB)) rtol = 1.0e-3
    fresh_hv = OptimizationBase.instantiate_function(
        optf, big.(x0), ad, big.(p0), 1; hv = true
    )
    HvBb = zeros(BigFloat, 2)
    fresh_hv.hv(HvBb, xB, vB)
    @test HvB ≈ HvBb

    # --- hv! with Float64 x and Float32 v (tangent type in the gate) ---
    Hv32 = zeros(Float32, 2)
    fi.hv(Hv32, Float32.(xt), Float32[1, 0])
    @test Float64.(Hv32) ≈ Hv_ref(xt, p0, [1.0, 0.0]) rtol = 1.0e-3
    Hv_mix = zeros(2)
    fi.hv(Hv_mix, xt, Float32[1, 0])
    @test Hv_mix ≈ Hv_ref(xt, p0, [1.0, 0.0]) rtol = 1.0e-3

    # --- OOP fg! / hess / fgh! / hv with SVector (different array type) ---
    xs = SVector{2}(xt)
    y, g = oop.fg(xs)
    @test y ≈ objp(xt, p0) rtol = 1.0e-6
    @test Vector(g) ≈ ∇xf(xt, p0) rtol = 1.0e-6
    fresh_fg = OptimizationBase.instantiate_function(
        OptimizationFunction{false}(objp, ad), xs, ad, p0, 0; fg = true
    )
    yb, gb = fresh_fg.fg(xs)
    @test y ≈ yb && g ≈ gb

    @test Matrix(oop.hess(xs)) ≈ Hess_ref(xt, p0) rtol = 1.0e-6
    y2, G2, H2 = oop.fgh(xs)
    @test y2 ≈ objp(xt, p0) rtol = 1.0e-6
    @test Vector(G2) ≈ ∇xf(xt, p0) rtol = 1.0e-6
    @test Matrix(H2) ≈ Hess_ref(xt, p0) rtol = 1.0e-6
    @test Vector(oop.hv(xs, SVector(1.0, 0.0))) ≈ Hv_ref(xt, p0, [1.0, 0.0]) rtol = 1.0e-6

    # --- IIP fgh! / fg! Float32 ---
    G32 = zeros(Float32, 2)
    H32c = zeros(Float32, 2, 2)
    y32 = fi.fgh(G32, H32c, Float32.(xt))
    @test Float64(y32) ≈ objp(xt, p0) rtol = 1.0e-3
    @test Float64.(G32) ≈ ∇xf(xt, p0) rtol = 1.0e-3
    @test Float64.(H32c) ≈ Hess_ref(xt, p0) rtol = 1.0e-3
    G32b = zeros(Float32, 2)
    fi.fg(G32b, Float32.(xt))
    @test Float64.(G32b) ≈ ∇xf(xt, p0) rtol = 1.0e-3

    # --- lag_h! with Float32 θ ---
    Hlag = zeros(Float32, 2, 2)
    fi.lag_h(Hlag, Float32.(xt), Float32(1), Float32[0.5])
    fresh_lag = OptimizationBase.instantiate_function(
        optf, Float32.(x0), ad, Float32.(p0), 1; lag_h = true
    )
    Hlagb = zeros(Float32, 2, 2)
    fresh_lag.lag_h(Hlagb, Float32.(xt), Float32(1), Float32[0.5])
    @test Hlag ≈ Hlagb
end
