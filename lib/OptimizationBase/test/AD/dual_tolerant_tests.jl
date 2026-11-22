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

using OptimizationBase, Test, ForwardDiff, FiniteDiff, ReverseDiff
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

function ∇xf(x, p)
    a, b = x
    q = p[2] * (b - a^2)
    return [2 * (a - p[1]) - 4a * q, 2q]
end
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

@testset "FiniteDiff constraint derivatives preserve mixed precision" begin
    objective(x, p) = sum(abs2, x)
    function constraint!(res, x, p)
        res[1] = p[1] + x[1]
        return nothing
    end
    ad = ADTypes.AutoFiniteDiff()
    for (x, p) in ((Float32[1, 2], [1.0e5]), ([1.0, 2.0], BigFloat[1.0e16]))
        @testset "Parameters of type $(eltype(p))" begin
            optf = OptimizationBase.instantiate_function(
                OptimizationFunction(objective, ad; cons = constraint!), x, ad, p, 1;
                cons_j = true, cons_vjp = true, cons_jvp = true
            )
            J = zeros(eltype(p), 1, 2)
            for params in (p, view(p, :))
                optf.cons_j(J, x, params)
                @test J ≈ [1 0] rtol = 2.0e-4
            end
            jvp, vjp = zeros(eltype(p), 1), zeros(eltype(p), 2)
            optf.cons_jvp(jvp, x, eltype(x)[2, 3])
            optf.cons_vjp(vjp, x, eltype(p)[2])
            @test jvp ≈ [2] rtol = 2.0e-4
            @test vjp ≈ [2, 0] rtol = 2.0e-4
        end
    end
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

function Hess_ref(x, p)
    a, b = x
    q = p[2]
    return [2 + 4q * (3a^2 - b) -4q * a; -4q * a 2q]
end
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

@testset "FiniteDiff fgh with prepared and call-time parameter types" begin
    objective(x, p) = p[1] * x[1]^2 + p[2] * x[1] * x[2]
    ad = ADTypes.AutoFiniteDiff()
    x, p = [1.0, 2.0], Float32[5, 7]
    fi = OptimizationBase.instantiate_function(
        OptimizationFunction(objective, ad), x, ad, p, 0; fgh = true
    )
    oop = OptimizationBase.instantiate_function(
        OptimizationFunction{false}(objective, ad), x, ad, p, 0; fgh = true
    )
    for params in (p, view(p, :))
        g, H = zeros(2), zeros(2, 2)
        fval = fi.fgh(g, H, x, params)
        fval_oop, g_oop, H_oop = oop.fgh(x, params)
        for (val, grad, hess) in ((fval, g, H), (fval_oop, g_oop, H_oop))
            @test val == 19
            @test grad ≈ [24, 7] rtol = 1.0e-5
            @test hess ≈ [10 7; 7 0] rtol = 1.0e-5
        end
    end
end

@testset "FiniteDiff Hessian at a domain boundary" begin
    function lower_objective(x, p)
        x[1] < 0 && throw(DomainError(x[1]))
        return (x[1] - 1)^2 + (x[2] - 1)^2
    end
    function upper_objective(x, p)
        x[1] > 1 && throw(DomainError(x[1]))
        return sum(abs2, x)
    end
    for (objective, x, ad) in (
            (lower_objective, zeros(2), ADTypes.AutoFiniteDiff()),
            (upper_objective, [1.0, 2.0], ADTypes.AutoFiniteDiff(; dir = -1, relstep = 0.0, absstep = 0.25)),
        )
        @testset "$(objective), in-place = $iip, parameters = $p" for iip in (true, false), p in (nothing, ())
            optf = OptimizationBase.instantiate_function(
                OptimizationFunction{iip}(objective, ad), x, ad, nothing, 0; h = true, fgh = true
            )
            G, H, H_fgh = zeros(2), zeros(2, 2), zeros(2, 2)
            if iip
                optf.hess(H, x, p)
                value = optf.fgh(G, H_fgh, x, p)
            else
                H = optf.hess(x, p)
                value, G, H_fgh = optf.fgh(x, p)
            end
            @test H ≈ [2.0 0.0; 0.0 2.0]
            @test H_fgh ≈ [2.0 0.0; 0.0 2.0]
            @test value == objective(x, p)
        end
    end
end

@testset "FiniteDiff gradient with an explicit backward step" begin
    objective(x, p) = sum(abs2, x)
    x = [1.0, 2.0]
    ad = ADTypes.AutoFiniteDiff(; dir = -1, relstep = 0.0, absstep = 0.25)
    @testset "in-place = $iip" for iip in (true, false)
        optf = OptimizationBase.instantiate_function(
            OptimizationFunction{iip}(objective, ad), x, ad, nothing, 0;
            g = true, fg = true, fgh = true
        )
        for p in (nothing, ())
            G, G_fg, G_fgh, H = zeros(2), zeros(2), zeros(2), zeros(2, 2)
            if iip
                optf.grad(G, x, p)
                value_fg = optf.fg(G_fg, x, p)
                value_fgh = optf.fgh(G_fgh, H, x, p)
            else
                G = optf.grad(x, p)
                value_fg, G_fg = optf.fg(x, p)
                value_fgh, G_fgh, H = optf.fgh(x, p)
            end
            @test G == G_fg == [1.75, 3.75]
            @test G_fgh == [1.75, 3.75]
            @test value_fgh == value_fg == 5.0
            @test H ≈ [2.0 0.0; 0.0 2.0]
        end
    end
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

    # --- lag_h! with Float32 θ (matrix and packed-vector IIP forms) ---
    Hlag = zeros(Float32, 2, 2)
    fi.lag_h(Hlag, Float32.(xt), Float32(1), Float32[0.5])
    fresh_lag = OptimizationBase.instantiate_function(
        optf, Float32.(x0), ad, Float32.(p0), 1; lag_h = true
    )
    Hlagb = zeros(Float32, 2, 2)
    fresh_lag.lag_h(Hlagb, Float32.(xt), Float32(1), Float32[0.5])
    @test Hlag ≈ Hlagb
    hlag = zeros(Float32, 3)
    fi.lag_h(hlag, Float32.(xt), Float32(1), Float32[0.5])
    hlagb = zeros(Float32, 3)
    fresh_lag.lag_h(hlagb, Float32.(xt), Float32(1), Float32[0.5])
    @test hlag ≈ hlagb

    # --- OOP lag_h with Float32 θ ---
    oop_lag = OptimizationBase.instantiate_function(
        OptimizationFunction{false}(objp, ad; cons = consp), x0, ad, p0, 1; lag_h = true
    )
    Hlag_oop = oop_lag.lag_h(Float32.(xt), Float32(1), Float32[0.5])
    fresh_oop_lag = OptimizationBase.instantiate_function(
        OptimizationFunction{false}(objp, ad; cons = consp), Float32.(x0), ad, Float32.(p0), 1;
        lag_h = true
    )
    Hlag_oop_b = fresh_oop_lag.lag_h(Float32.(xt), Float32(1), Float32[0.5])
    @test Hlag_oop ≈ Hlag_oop_b
end

@testset "lag_h at the call-time p, σ = 0, and foreign σ/λ types" begin
    # Both constraints depend on `p`, so a constraint term evaluated at the construction `p`
    # gives a different Hessian.
    lcons!(res, x, p) = (res[1] = p[1] * x[1]^2 + x[2]^2; res[2] = p[2] * x[1] * x[2]; nothing)
    lcons(x, p) = [p[1] * x[1]^2 + x[2]^2, p[2] * x[1] * x[2]]
    lag_ref(θ, σ, λ, p) = σ * Hess_ref(Float64.(θ), p) +
        [2λ[1] * p[1] λ[2] * p[2]; λ[2] * p[2] 2λ[1]]
    packed(H) = [H[i, j] for i in 1:size(H, 1) for j in 1:i]
    λ = [0.5, -1.5]
    xf, λf, p1f = Float32.(xt), Float32.(λ), Float32.(p1)
    cases = (
        (xt, 1.0, λ, (p1,)),
        (xt, 1.0, λ, (p1f,)),
        (xf, 1.0f0, λf, (p1f,)),
        (xt, 0.0, λ, ()),
        (xt, 0.0, λ, (p1,)),
        (xf, 0.0f0, λf, ()),
        (xf, 0.0f0, λf, (p1f,)),
        (xt, 2, λ, ()),
        (xt, 1.0, λf, ()),
        (xt, 1.0, view([λ; 9.0], 1:2), ()),
    )
    for ad in (ADTypes.AutoForwardDiff(), ADTypes.AutoReverseDiff())
        fi = OptimizationBase.instantiate_function(
            OptimizationFunction(objp, ad; cons = lcons!), x0, ad, p0, 2; lag_h = true
        )
        oop = OptimizationBase.instantiate_function(
            OptimizationFunction{false}(objp, ad; cons = lcons), x0, ad, p0, 2; lag_h = true
        )
        for (θ, σ, λc, pargs) in cases
            ref = lag_ref(θ, σ, λc, isempty(pargs) ? p0 : only(pargs))
            rtol = eltype(θ) == Float32 ? 1.0e-5 : 1.0e-10
            H = zeros(eltype(θ), 2, 2)
            fi.lag_h(H, θ, σ, λc, pargs...)
            @test H ≈ ref rtol = rtol
            h = zeros(eltype(θ), 3)
            fi.lag_h(h, θ, σ, λc, pargs...)
            @test h ≈ packed(ref) rtol = rtol
            @test oop.lag_h(θ, σ, λc, pargs...) ≈ ref rtol = rtol
        end
    end

    @testset "Constraint AD fallback without per-constraint preparations" begin
        ad = ADTypes.AutoForwardDiff()
        user_cons_h!(H, x, p) = (H[1] .= [2p[1] 0; 0 2]; H[2] .= [0 p[2]; p[2] 0]; nothing)
        fi = OptimizationBase.instantiate_function(
            OptimizationFunction(objp, ad; cons = lcons!, cons_h = user_cons_h!), x0, ad, p0, 2;
            lag_h = true
        )
        H = zeros(2, 2)
        fi.lag_h(H, xt, 0.0, λ)
        @test H ≈ lag_ref(xt, 0.0, λ, p0) rtol = 1.0e-10
    end
end

@testset "Lagrangian constraints use the state element type" begin
    ad = ADTypes.AutoForwardDiff()
    objective(x, p) = sum(abs2, x)
    function constraint!(res, x, p)
        res[1] = abs2(p[1]) * x[1]^2
        return nothing
    end
    function same_type_constraint!(res::AbstractVector{T}, x::AbstractVector{T}, p) where {T}
        return constraint!(res, x, p)
    end
    for (cons!, x, p, diagonal) in (
            (constraint!, [1.0, 2.0], [2.0 + im], 12),
            (same_type_constraint!, Float32[1, 2], [2.0], 10),
        )
        @testset "Parameters of type $(eltype(p))" begin
            fi = OptimizationBase.instantiate_function(
                OptimizationFunction(objective, ad; cons = cons!), x, ad, p, 1; lag_h = true
            )
            H, packed = zeros(eltype(x), 2, 2), zeros(eltype(x), 3)
            σ, λ = one(eltype(x)), ones(eltype(x), 1)
            fi.lag_h(H, x, σ, λ)
            fi.lag_h(packed, x, σ, λ)
            @test H == [diagonal 0; 0 2]
            @test packed == [diagonal, 0, 2]
        end
    end
end

@testset "Zero-weight Lagrangian skips the objective" begin
    objective(x, p) = log(x[1]) + x[2]^2
    ad = ADTypes.AutoForwardDiff()
    x = [-1.0, 2.0]
    fi = OptimizationBase.instantiate_function(
        OptimizationFunction(objective, ad; cons = consp!), [1.0, 2.0], ad, p0, 1; lag_h = true
    )
    for p in (p0, p1)
        H, packed = zeros(2, 2), zeros(3)
        fi.lag_h(H, x, 0.0, [1.0], p)
        fi.lag_h(packed, x, 0.0, [1.0], p)
        @test H == [2p[1] 0; 0 2]
        @test packed == [2p[1], 0, 2]
    end
end
