module ConvexOptimization

using Reexport
@reexport using SciMLBase
using SciMLBase: ConvexOptimizationProblem, OptimizationSolution,
    OptimizationFunction, AbstractOptimizationCache, AbstractOptimizationAlgorithm,
    NullParameters, ReturnCode
import MathOptInterface as MOI
import Symbolics
using Symbolics: variable, unwrap, linear_expansion
import SymbolicAnalysis
using SymbolicAnalysis: analyze
import SymbolicUtils
using LinearAlgebra

"""
    ConeConstraint(g, set)

One convex cone constraint of a [`ConvexOptimizationProblem`](@ref). `g(u, p)`
returns the affine map whose image must lie in the MathOptInterface vector cone
`set` (`MOI.Zeros`, `MOI.Nonnegatives`, `MOI.Nonpositives`, `MOI.SecondOrderCone`,
…). The output length of `g` must equal `MOI.dimension(set)`.

The backend traces `g` on its own symbolic variables, so each `ConeConstraint`
maps to exactly one MOI constraint and therefore one entry of the returned
`OptimizationSolution.dual`, in the order the constraints are given.
Because the cone is named explicitly, the returned dual is already expressed in
the user's variables (no sign remap): `>=` → `MOI.Nonnegatives`, `<=` →
`MOI.Nonpositives`, `==` → `MOI.Zeros`.

`set` is a *value*, not a function of `p`: it is read once when the cache is built
and is frozen for the life of that cache. A set carrying numeric data derived from
the parameters (`MOI.PowerCone(p[1])`) will **not** be updated by
[`SciMLBase.reinit!`](@ref); rebuild with `solve(remake(prob; p = …), alg)` instead.
"""
struct ConeConstraint{G, S <: MOI.AbstractVectorSet}
    g::G
    set::S
end

abstract type AbstractConvexOptAlgorithm <: AbstractOptimizationAlgorithm end

"""
    ConvexMOI(optimizer_constructor = Clarabel.Optimizer)

Conic backend: certify convexity with SymbolicAnalysis, lower the objective and
each `ConeConstraint` to a MathOptInterface cone, and solve with
`optimizer_constructor`.

The objective may be affine or contain atoms that are lowered through their
epigraph: `minimize norm(A*u - b, 2)` introduces an epigraph variable `τ` with
`(τ, A*u - b) ∈ SecondOrderCone` and minimizes `τ`. Atoms may also nest inside
one another — `norm(A*u - b)^2`, `exp(norm(u))`, `max(norm(u), 1)`,
`sum(abs.(u))^2`, `norm([exp(u[1]), u[2]])` — in which case each inner atom
gets its own epigraph variable and the outer atom's argument becomes affine in
it. A nested composition is accepted only when the *original* expression is
DCP-certified by `SymbolicAnalysis.analyze`; sign-dependent monotonicity is
what makes e.g. `norm([exp(u[1]), u[2]])` convex but `abs2(max(u[1], u[2]))`
not, so the latter is rejected. Supported atoms are

  - `norm(w, p)` for `p = 1, 2, Inf` (`NormOneCone`, `SecondOrderCone`,
    `NormInfinityCone`), with `w` an array expression in `u`;
  - `abs(w)` for scalar affine `w` (`NormOneCone`);
  - `max(w1, w2, …)`/`maximum(w)` of affine scalars or a vector affine `w`
    (`Nonnegatives` epigraph), and their concave mirrors `min`/`minimum`
    (hypograph: valid under `MaxSense`, or entering negatively under
    `MinSense` as `-min(…)`);
  - `sum(abs.(w))` and `maximum(abs.(w))` for vector affine `w` — the l1 and
    linf norms (`NormOneCone`, `NormInfinityCone`);
  - `exp(w)` and `log(w)` for scalar affine `w` (`ExponentialCone`); `log` is
    concave, so it is bounded below and must enter the objective negatively
    (e.g. a `-log` barrier);
  - `abs2(w)` and `w^2` for scalar affine `w`; `sum(abs2.(w))`, `sum(w .^ 2)`,
    `sum(abs2, w)` and the self-product `v' * v` for vector affine `w`/`v`
    (e.g. `A*u - b`); and the quadratic forms `quad_form(u, P)` (the
    SymbolicAnalysis atom) and `u' * P * u` with `P` a constant positive
    semidefinite matrix (`MOI.RotatedSecondOrderCone`: `‖w‖² <= τ` is
    `(τ, 1/2, w)`, and `P` is factored as `LᵀL` once at build time). A `P`
    that is not positive semidefinite or that depends on `p` is rejected. A
    square whose argument contains no optimization variable, e.g. `p[1]^2`,
    is a `p`-dependent constant — not an atom — and may enter the objective
    with either sign.

Every "affine `w`" above means affine in `u` *and* in the epigraph variables
of nested atoms: an argument element that is itself a supported atom (or an
affine combination of atoms) is lowered recursively, innermost first.
Compositions are only accepted when the composed expression is convex under
DCP monotonicity — `exp` of a convex atom, `max`/`maximum`/`norm`/`abs` of
convex nonnegative atoms, `abs2`/`^2` of a convex nonnegative atom, `min`/
`minimum`/`log` of concave atoms under `MaxSense`, and affine combinations of
all of those.

Keep a `norm` argument an array expression built from `u` (`A*u - b`, `u .- c`,
or a literal `[e1, e2, …]` whose elements may be atoms); a `norm` computed by
hand as `sqrt(sum(w .^ 2))` is a different, unsupported expression.

Parameters are first-class: see [`SciMLBase.reinit!`](@ref) for re-solving at a new
`p` without re-running the symbolic canonicalization. Because that certificate
is built once and reused at every `p`, a nested composition containing
parameters is accepted only when it is convex for *every* `p` — e.g.
`norm(A*u - p)^2` is accepted but `(exp(u[1]) + p[1])^2` is refused, since its
curvature depends on the sign of `exp(u[1]) + p[1]`. A subexpression involving
`p` alone is certified as a constant of unknown sign (a literal square such as
`p[1]^2` keeps its nonnegative sign), so `norm(u)^2 - p[1]^2` and
`p[1]^2 * norm(u)^2` are accepted without requiring joint convexity in
`(u, p)`. An objective that is only convex at some parameter values can be
solved by re-canonicalizing with `solve(remake(prob; p = θ), alg)` at each `θ`.
"""
struct ConvexMOI{O} <: AbstractConvexOptAlgorithm
    optimizer_constructor::O
end

SciMLBase.allowsbounds(::AbstractConvexOptAlgorithm) = true
SciMLBase.allowsconstraints(::AbstractConvexOptAlgorithm) = true

# Must be <: AbstractOptimizationCache (build_convex_solution requires it) and
# carry real `f`/`p` fields for the solution's SymbolicIndexingInterface glue.
# No `reinit_cache` field: it would reroute getproperty(:u0/:p) *and* make
# `SciMLBase.has_reinit` true, so the generic `reinit!` would set `p` and leave the
# MOI model stale. Without it that method throws, and the specialization below is
# the only way to re-solve.
mutable struct ConvexOptimizationCache{F, U, P, A, AR, D, MOD, XV, CR, TR} <:
    AbstractOptimizationCache
    f::F
    u0::U
    p::P                 # Vector{Float64} or NullParameters; length fixed at `init`
    alg::A
    analysis::AR
    dpp::D               # DPPData: numeric cone data as an affine function of `p`
    model::MOD           # lowered MOI model
    xvars::XV            # Vector{MOI.VariableIndex}: user u -> MOI variables
    conrefs::CR          # Vector{MOI.ConstraintIndex}, 1:1 with prob.constraints
    atomrefs::TR         # cones introduced by epigraph lowering; never in `sol.dual`
end

# `solve(prob, alg)` routes through CommonSolve: solve = solve! ∘ init. Neither
# `init` nor `solve!` is inherited here (no OptimizationBase in the dep tree), so
# both thin methods are defined explicitly.
function SciMLBase.init(
        prob::ConvexOptimizationProblem,
        alg::AbstractConvexOptAlgorithm, args...; kwargs...
    )
    return SciMLBase.__init(prob, alg, args...; prob.kwargs..., kwargs...)
end
SciMLBase.solve!(cache::ConvexOptimizationCache) = SciMLBase.__solve(cache)

function SciMLBase.__init(
        prob::ConvexOptimizationProblem,
        alg::AbstractConvexOptAlgorithm, args...; kwargs...
    )
    # One trace of `f`/`g` feeds certification, the parameter-affine extraction and
    # the lowering: the epigraph variables minted here must be the same ones the
    # numeric tensors and the MOI model are built from.
    tr = _trace_problem(prob)
    analysis = certify_convex(prob, tr)
    dpp = _dpp_extract(prob, tr)
    model = MOI.instantiate(alg.optimizer_constructor; with_bridge_type = Float64)
    xvars, conrefs, atomrefs = _build_moi!(model, dpp, _theta(prob.p))
    return ConvexOptimizationCache(
        prob.f, prob.u0, _cachep(prob.p), alg, analysis, dpp,
        model, xvars, conrefs, atomrefs
    )
end

function SciMLBase.__solve(cache::ConvexOptimizationCache)
    model = cache.model
    MOI.optimize!(model)
    ret = _moi_status_to_retcode(MOI.get(model, MOI.TerminationStatus()))
    if MOI.get(model, MOI.ResultCount()) >= 1
        u = MOI.get(model, MOI.VariablePrimal(), cache.xvars)
        objective = MOI.get(model, MOI.ObjectiveValue())
    else
        u = fill(NaN, length(cache.xvars))
        objective = NaN
    end
    # A dual is reported only when it *is* one. `DualStatus` is
    # `INFEASIBILITY_CERTIFICATE`/`INFEASIBLE_POINT`/`OTHER_RESULT_STATUS` on a
    # failed solve (never `NO_SOLUTION` while `ResultCount() >= 1`), and returning
    # those as `sol.dual` would hand back a vector of the right shape that is not a
    # dual solution. NaNs keep `sol.dual` 1:1 with the user constraints and keep its
    # type `Vector{Vector{Float64}}` unconditionally (`build_convex_solution`
    # defaults to `calculate_dual = Val(true)`, which cannot convert `nothing`).
    dual = if MOI.get(model, MOI.ResultCount()) >= 1 && MOI.get(model, MOI.DualStatus()) in
            (MOI.FEASIBLE_POINT, MOI.NEARLY_FEASIBLE_POINT)
        [MOI.get(model, MOI.ConstraintDual(), c) for c in cache.conrefs]
    else
        [fill(NaN, MOI.dimension(s)) for s in cache.dpp.consets]
    end
    return SciMLBase.build_convex_solution(
        cache, cache.alg, u, objective;
        dual = dual, retcode = ret, original = model,
        stats = SciMLBase.OptimizationStats()
    )
end

"""
    SciMLBase.reinit!(cache::ConvexOptimizationCache; p = missing, u0 = missing)

Re-solve the cached problem at a new parameter vector `p` without re-running any
symbolic work. The trace, the convexity certificate and the parameter-affine
extraction all happened once in `init`; this only evaluates

    c(p) = c0 + C*p,  d(p) = d0 + dP⋅p,  b_k(p) = b0_k + B_k*p

and rebuilds the MOI model from those numbers. Returns the cache, which callers
must reassign:

```julia
cache = init(prob, ConvexMOI(Clarabel.Optimizer))
sol1  = solve!(cache)
cache = reinit!(cache; p = [2.0, 0.5])
sol2  = solve!(cache)
```

`u0` is accepted and stored for API uniformity but does not affect the answer: a
conic solve is global, and Clarabel does not support `MOI.VariablePrimalStart`.

!!! warning
    `reinit!` mutates the cache in place. A previously returned solution's `u`,
    `objective` and `dual` are snapshots and stay correct, but its `original`
    field aliases the cache's MOI model and its `cache.p` (hence
    `SciMLBase.get_p(sol)`) reports the *current* `p`. Pair a dual with the `p` it
    was solved at before calling `reinit!`. Anything a `ConeConstraint`'s `set`
    carries is frozen at `init` and is not refreshed here.
"""
function SciMLBase.reinit!(
        cache::ConvexOptimizationCache; p = missing, u0 = missing,
        interpret_symbolicmap = true
    )
    if u0 !== missing
        eltype(u0) <: Pair && error(
            "`reinit!` on a `ConvexOptimizationCache` does not support symbolic maps; " *
                "pass `u0` as a vector of values."
        )
        length(u0) == length(cache.u0) || error(
            "`reinit!` got a `u0` of length $(length(u0)), but this cache was " *
                "canonicalized for $(length(cache.u0)) optimization variables. The " *
                "variable count is fixed at `init`; use " *
                "`solve(remake(prob; u0 = …), alg)` to canonicalize a new problem."
        )
        cache.u0 = u0
    end
    p === missing && return cache
    _validate_p(cache, p)
    θ = Float64.(p)
    # Build first, assign second: a rejected `p` (a sign flip that invalidates an
    # epigraph lowering) must leave the cache exactly as it was.
    xvars, conrefs, atomrefs = _build_moi!(cache.model, cache.dpp, θ)
    cache.xvars, cache.conrefs, cache.atomrefs = xvars, conrefs, atomrefs
    cache.p = θ
    return cache
end

function _validate_p(cache::ConvexOptimizationCache, p)
    eltype(p) <: Pair && error(
        "`reinit!` on a `ConvexOptimizationCache` does not support symbolic maps " *
            "(`p = [sym => val, …]`); this problem has no symbolic origin. Pass `p` as a " *
            "vector of values in the order of the original `p`."
    )
    cache.p isa NullParameters && error(
        "`reinit!` got `p`, but this problem was canonicalized with `NullParameters`: " *
            "there are no parameters in the cached problem data to update. Build the " *
            "problem with a parameter vector, or use `solve(remake(prob; p = …), alg)`."
    )
    p isa AbstractVector || error(
        "`reinit!` expects `p` as an `AbstractVector`; got a `$(typeof(p))`."
    )
    length(p) == cache.dpp.m || error(
        "`reinit!` got a `p` of length $(length(p)), but this cache was canonicalized " *
            "for $(cache.dpp.m) parameters. The parameter count is fixed at `init`; use " *
            "`solve(remake(prob; p = …), alg)` to canonicalize a new problem."
    )
    all(isfinite, p) || error(
        "`reinit!` got a non-finite entry in `p` ($(p)); the cone program's data would " *
            "not be well defined."
    )
    return nothing
end

_theta(p) = p isa NullParameters ? Float64[] : Float64.(p)
_cachep(p) = p isa NullParameters ? p : Float64.(p)

certify_convex(prob::ConvexOptimizationProblem) = certify_convex(prob, _trace_problem(prob))

# Certification runs on the *lowered* objective (atoms already replaced by their
# epigraph variables). Each atom's argument is proven affine separately, by
# `linear_expansion` in `_dpp_extract`, which is a stronger check than DCP.
#
# `analyze` cannot certify a flat parametric objective (it has no constant-vs-
# variable distinction, and substituting one θ would not cover `reinit!`), so
# the structural predicate in `_dpp_extract` is the certificate instead.
function certify_convex(prob::ConvexOptimizationProblem, tr)
    tr.nested && return _certify_nested(prob, tr)
    isempty(tr.params) || return nothing
    obj_res = analyze(unwrap(tr.objl))
    _check_obj_curvature(prob, obj_res, tr)
    cons_res = _certify_constraints(prob, tr)
    return (; objective = obj_res, constraints = cons_res)
end

function _check_obj_curvature(prob, res, tr; suffix = "")
    _curvature_admits(res, prob.sense) || error(
        "Objective is not certified convex for $(prob.sense): curvature = " *
            "$(res.curvature)." * suffix *
            " Route to a general OptimizationProblem/NLP solver." * _norm_hint(tr.obj)
    )
    return nothing
end

_curvature_admits(res, sense) =
    sense === SciMLBase.MaxSense ?
    res.curvature in (SymbolicAnalysis.Concave, SymbolicAnalysis.Affine) :
    res.curvature in (SymbolicAnalysis.Convex, SymbolicAnalysis.Affine)

# The outer atom's pull on an inner epigraph variable is DCP's sign-dependent
# monotonicity, so `analyze` runs on the original expression — via exact
# rewrites — with parameters symbolic so the certificate holds at every θ.
function _certify_nested(prob, tr)
    res = _nested_curvature(prob, tr)
    _check_obj_curvature(
        prob, res, tr;
        suffix = isempty(tr.params) ? "" :
            " Parameters are kept symbolic in the certificate, so the curvature " *
            "may also depend on `p`; if it is only convex at some parameter " *
            "values, re-canonicalize with `solve(remake(prob; p = …), alg)`."
    )
    cons_res = isempty(tr.params) ? _certify_constraints(prob, tr) : nothing
    return (; objective = res, constraints = cons_res)
end

# `analyze` can throw instead of reporting `UnknownCurvature` (e.g. a `c' * x`
# term carries a `1×1` shape that trips `propagate_sign`); a throw means the
# same thing as `UnknownCurvature` — try the next equivalent form.
_try_analyze(ex) = try
    analyze(ex)
catch e
    e isa InterruptException && rethrow()
    (;
        curvature = SymbolicAnalysis.UnknownCurvature,
        sign = SymbolicAnalysis.AnySign,
    )
end

const THC_BASE = Symbol("##θc")

# `analyze` has no constants: a θ-only subterm as a fresh symbol is affine
# (unknown sign); a second form keeps subterms provably positive for every θ
# so enclosing monotonicity rules can use their sign.
function _theta_constant_forms(ex, optset)
    i = Ref(0)
    subst(keep) = SymbolicUtils.Rewriters.Prewalk() do t
        _is_theta_subterm(t, optset) || return nothing
        keep(t) && return t
        return variable(THC_BASE, (i[] += 1))
    end(ex)
    return subst(_ -> false),
        subst(t -> _try_analyze(t).sign === SymbolicAnalysis.Positive)
end

# A θ-only factor of provable sign (for every θ) drops out of a product —
# positive scales the curvature, negative flips it — while an unknown-sign
# factor stays and the multiplication rule refuses the product.
function _strip_theta_products(ex, optset)
    return SymbolicUtils.Rewriters.Postwalk() do t
        (Symbolics.iscall(t) && Symbolics.operation(t) === (*)) || return t
        kept = Any[]
        neg = false
        for a in Symbolics.arguments(t)
            if _is_theta_subterm(a, optset)
                s = _try_analyze(a).sign
                s === SymbolicAnalysis.Positive && continue
                s === SymbolicAnalysis.Negative && (neg = !neg; continue)
            end
            push!(kept, a)
        end
        r = isempty(kept) ? 1 :
            length(kept) == 1 ? kept[1] :
            Symbolics.SymbolicUtils.maketerm(typeof(t), (*), kept, Symbolics.metadata(t))
        return neg ?
            Symbolics.SymbolicUtils.maketerm(typeof(t), (*), Any[-1, r], Symbolics.metadata(t)) :
            r
    end(ex)
end

_is_theta_subterm(t, optset) =
    t isa Symbolics.SymbolicUtils.BasicSymbolic &&
    !isempty(Symbolics.get_variables(t)) &&
    !_has_optvar(t, optset)

function _nested_curvature(prob, tr)
    ex = unwrap(tr.obj)
    # Lifted `##psq` variables stand for `arg^2`; restoring the expression lets
    # `analyze` see that the term is nonnegative for every θ.
    for (s, a) in zip(tr.psqvars, tr.psqargs)
        ex = Symbolics.substitute(ex, Dict{Any, Any}(unwrap(s) => unwrap(a)^2))
    end
    forms = if isempty(tr.params)
        (ex,)
    else
        optset = Set(unwrap.(tr.cols))
        map(f -> _strip_theta_products(f, optset), _theta_constant_forms(ex, optset))
    end
    res = nothing
    for e in forms
        res = _analyze_stages(e, prob.sense)
        _curvature_admits(res, prob.sense) && return res
    end
    return res
end

function _analyze_stages(ex, sense)
    res = _try_analyze(ex)
    _curvature_admits(res, sense) && return res
    # `analyze` misses elementwise monotonicity in vector atom arguments.
    ex = try
        _dcp_reassociate(ex)
    catch e
        e isa InterruptException && rethrow()
        ex
    end
    res = _try_analyze(ex)
    _curvature_admits(res, sense) && return res
    # `c' * x` terms carry a `1×1` symtype that trips `propagate_sign`.
    ex = _expand_scalar_products(ex)
    return _try_analyze(ex)
end

function _dcp_reassociate(ex)
    return SymbolicUtils.Rewriters.Postwalk(_reassoc_node)(ex)
end

function _reassoc_node(x)
    Symbolics.iscall(x) || return x
    op = Symbolics.operation(x)
    args = Symbolics.arguments(x)
    if op === LinearAlgebra.norm
        length(args) in (1, 2) || return x
        p = length(args) == 1 ? 2 : Symbolics.value(args[2])
        p isa Number || return x
        es = _try_scalarize(args[1])
        es === nothing && return x
        p == 1 && return foldl(+, Any[abs(e) for e in es]; init = 0.0)
        p == 2 && return length(es) == 1 ? abs(only(es)) : foldl(hypot, es)
        isinf(p) && p > 0 && return foldl(max, Any[abs(e) for e in es])
        return x
    elseif op isa SymbolicUtils.Mapreducer
        r = _mapreducer_reassoc(op, args)
        return r === nothing ? x : r
    elseif op === (*)
        parts = _quad_form_parts(x)
        parts === nothing && return x
        lv = _quad_form_Lv(parts.v, parts.mid)
        return foldl(+, Any[abs2(e) for e in lv]; init = 0.0)
    end
    return x
end

function _mapreducer_reassoc(op::SymbolicUtils.Mapreducer, args)
    length(args) == 1 || return nothing
    arg = args[1]
    if op.f === abs2 && op.reduce === Base.add_sum
        es = _try_scalarize(arg)
        es === nothing && return nothing
        return foldl(+, Any[abs2(e) for e in es]; init = 0.0)
    end
    op.f === identity || return nothing
    if !(Symbolics.iscall(arg) && Symbolics.operation(arg) === broadcast)
        (op.reduce === max || op.reduce === min) || return nothing
        es = _try_scalarize(arg)
        es === nothing && return nothing
        return op.reduce === max ? foldl(max, es) : foldl(min, es)
    end
    bargs = Symbolics.arguments(arg)
    bf = Symbolics.value(unwrap(bargs[1]))
    if bf === abs && length(bargs) == 2
        es = _try_scalarize(bargs[2])
        es === nothing && return nothing
        op.reduce === Base.add_sum &&
            return foldl(+, Any[abs(e) for e in es]; init = 0.0)
        op.reduce === max && return foldl(max, Any[abs(e) for e in es])
        return nothing
    end
    op.reduce === Base.add_sum || return nothing
    sq = bf === abs2 && length(bargs) == 2
    if !sq && bf === (^) && length(bargs) == 3 && _is_square_exp(bargs[3])
        sq = true
    end
    sq || return nothing
    es = _try_scalarize(bargs[2])
    es === nothing && return nothing
    return foldl(+, Any[abs2(e) for e in es]; init = 0.0)
end

function _try_scalarize(a)
    s = try
        Symbolics.scalarize(Symbolics.wrap(a))
    catch e
        e isa InterruptException && rethrow()
        return nothing
    end
    s isa AbstractArray || return Any[unwrap(s)]
    return Any[unwrap(e) for e in vec(collect(s))]
end

# MVP: constraints are affine-in-cone, so every output component must be Affine.
function _certify_constraints(prob, tr)
    tr.consvals === nothing && return nothing
    res = []
    for (con, gvals) in zip(prob.constraints, tr.consvals)
        cres = analyze.(unwrap.(gvals))
        all(r -> r.curvature == SymbolicAnalysis.Affine, cres) || error(
            "This backend supports affine-in-cone constraints only; got " *
                "curvatures $(getproperty.(cres, :curvature)) for cone $(con.set)."
        )
        push!(res, cres)
    end
    return res
end

# ---------------------------------------------------------------------------
# Disciplined parametrized programming: problem data affine in `p`
# ---------------------------------------------------------------------------

"""
    DPPData

Internal: the canonicalized cone program's numeric data, as an affine function of
the parameter vector `θ = p`:

    objective    c(θ) = c0 + C*θ   over `z = [u; τ]`,   constant  d(θ) = d0 + dP⋅θ
    constraint k A_k z + b_k(θ),   b_k(θ) = conb0[k]  + conB[k]*θ    ∈ consets[k]
    atom cone j M_j z + e_j(θ),    e_j(θ) = atomb0[j] + atomB[j]*θ   ∈ atomsets[j]

`A_k`, `M_j`, every cone set, `lb`/`ub` and `sense` are θ-free, so the MOI model
rebuilt at any θ has identical variable and constraint numbering — which is what
lets `conrefs` stay 1:1 with `prob.constraints` across a `reinit!`. Do not
introduce θ-dependent emission of any variable or cone.

Parameter-only squares (`p[1]^2`, `abs2(p[1])`) are lifted to `length(psqs)`
implicit extra columns appended after the `m` user parameters and evaluated
numerically from `psyms`/`psqs` at every θ, so a `-p[1]^2` term can stay a
θ-dependent constant instead of forcing an epigraph whose sign guard would
reject it. `m` counts only the user parameters.
"""
struct DPPData{CS, AS, SE}
    n::Int                          # user variables
    m::Int                          # user parameters (reinit! validates against this)
    c0::Vector{Float64}             # length n + ntau
    C::Matrix{Float64}              # (n + ntau) × (m + length(psqs))
    d0::Float64
    dP::Vector{Float64}             # length m + length(psqs)
    psqfns::Vector                  # compiled evaluators θ -> lifted square values
    psqs::Vector                    # arguments of lifted parameter-only squares
    conA::Vector{Matrix{Float64}}
    conb0::Vector{Vector{Float64}}
    conB::Vector{Matrix{Float64}}
    consets::CS
    atomA::Vector{Matrix{Float64}}
    atomb0::Vector{Vector{Float64}}
    atomB::Vector{Matrix{Float64}}
    atomsets::AS
    atomdirs::Vector{Int}
    lb::Vector{Float64}
    ub::Vector{Float64}
    sense::SE
end

const _PNOTE = " (`var\"##pᵢ\"` denotes `p[i]`; `var\"##τₖ\"` is the epigraph variable of atom k.)"

_isnum(e) = Symbolics.value(e) isa Number
_floatmat(A) = Float64[_tofloat(A[i, j]) for i in axes(A, 1), j in axes(A, 2)]
_floatvec(b) = Float64[_tofloat(e) for e in b]

# `linear_expansion` is greedy: it reports `islin = true` for `x[1]*x[2]` with the
# coefficient `x[2]`, and for `ifelse(x[1] > 0, …)` it hides the variable in the
# constant. So `islin` alone proves nothing; an extracted coefficient *or constant*
# that still mentions an optimization variable is the real non-affine case. This
# check is what replaces `analyze` as the convexity gate on the parametric path.
function _check_no_optvars(es, paramset, tauset, what)
    for e in es, v in Symbolics.get_variables(e)
        unwrap(v) in paramset && continue
        unwrap(v) in tauset && error(
            "$what scales a lowered atom by a non-constant expression: the expansion " *
                "left the coefficient `$e`, which contains `$v`, the epigraph variable of " *
                "an atom. An atom may be scaled only by a quantity that is affine in `p`. " *
                "If `$v` came from lowering a function of `p` alone (as in " *
                "`exp(p[1])*u[1]`), pass that quantity as its own parameter instead." * _PNOTE
        )
        error(
            "$what is not affine in the optimization variables: the expansion left the " *
                "coefficient `$e`, which still contains the optimization variable `$v`. " *
                "Products or nonlinear functions of the optimization variables are not " *
                "disciplined-convex here; route to a general OptimizationProblem/NLP solver."
        )
    end
    return nothing
end

# First entry that is not a plain number, or `nothing`. Only meaningful when the
# expansion succeeded: `islin == false` leaves #undef entries behind.
function _nonnumeric(xss...)
    for xs in xss
        i = findfirst(!_isnum, xs)
        i === nothing || return xs[i]
    end
    return nothing
end

# Second-stage expansion: `es == Bp*θ + b0` with both tensors numeric. `islin` is
# again not sufficient (`p₁*p₂` expands with `islin = true` and coefficient `p₂`).
function _theta_affine(es, params, what)
    Bp, b0, islin = linear_expansion(collect(es), params)
    if !islin || !all(_isnum, Bp) || !all(_isnum, b0)
        bad = islin ? _nonnumeric(Bp, b0) : nothing
        witness = bad === nothing ? "" : ": the expansion left the non-numeric term `$bad`"
        error(
            "$what is not affine in the parameters `p`$witness. Disciplined " *
                "parametrized programming requires the cone program's data to be affine " *
                "in `p`; rewrite the nonlinear function of `p` as an extra parameter, or " *
                "use `solve(remake(prob; p = …), alg)` to re-canonicalize at each `p`." *
                _PNOTE
        )
    end
    return _floatvec(b0), _floatmat(Bp)
end

# One cone block: affine in `z` with a θ-free matrix and a θ-affine constant.
function _dpp_block(rows, cols, params, paramset, tauset, what, nonaffine_msg)
    A, b, islin = linear_expansion(collect(rows), cols)
    islin || error(nonaffine_msg)                     # `A` has #undef entries if false
    _check_no_optvars(A, paramset, tauset, what)
    _check_no_optvars(b, paramset, tauset, what)
    all(_isnum, A) || error(
        "$what has the parameter-dependent coefficient `$(_nonnumeric(A))`. " *
            "This problem is convex at every `p`, but its cone *matrix* moves with `p`, so it " *
            "cannot be canonicalized once and re-solved from cached data. Rewrite so that " *
            "parameters enter additively (`A*u - p` rather than `p*u`), or use " *
            "`solve(remake(prob; p = …), alg)` to re-canonicalize at each `p`." * _PNOTE
    )
    b0, Bp = _theta_affine(b, params, "$what's constant term")
    return _floatmat(A), b0, Bp
end

function _dpp_extract(prob::ConvexOptimizationProblem, tr)
    n = length(prob.u0)
    cols, allcols = tr.cols, tr.allcols
    params = vcat(tr.params, tr.psqvars)
    paramset = Set(unwrap.(params))
    tauset = Set(unwrap.(tr.taus))
    m = length(tr.params)

    conA, conb0, conB = Matrix{Float64}[], Vector{Float64}[], Matrix{Float64}[]
    consets = prob.constraints === nothing ? MOI.AbstractVectorSet[] :
        [con.set for con in prob.constraints]
    if prob.constraints !== nothing
        for (k, (con, gvals)) in enumerate(zip(prob.constraints, tr.consvals))
            A, b0, Bp = _dpp_block(
                gvals, cols, params, paramset, tauset, "Constraint $k ($(con.set))",
                "Constraint $(con.set) is not affine in the variables."
            )
            push!(conA, A); push!(conb0, b0); push!(conB, Bp)
        end
    end
    @assert length(conA) == length(consets)

    atomA, atomb0, atomB = Matrix{Float64}[], Vector{Float64}[], Matrix{Float64}[]
    for (j, at) in enumerate(tr.atoms)
        A, b0, Bp = _dpp_block(
            at.rows, allcols, params, paramset, tauset,
            "The argument of atom $j ($(at.set))",
            "The argument of an atom in the objective must be affine in the " *
                "optimization variables."
        )
        push!(atomA, A); push!(atomb0, b0); push!(atomB, Bp)
    end

    Ao, bo, olin = linear_expansion(_asvec(tr.objl), allcols)
    olin || error(
        "This backend requires an objective that is affine in the optimization " *
            "variables and in the epigraph variables of its lowered atoms."
    )
    _check_no_optvars(Ao, paramset, tauset, "The objective")
    _check_no_optvars(bo, paramset, tauset, "The objective")
    # The objective's coefficients are the one place a parameter may multiply a
    # column: `c(θ)` is linear data, and a linear objective is convex at every θ.
    c0, C = _theta_affine(vec(Ao), params, "An objective coefficient")
    d0v, dPm = _theta_affine(bo, params, "The objective's constant term")

    lb = prob.lb === nothing ? fill(-Inf, n) : Float64.(collect(prob.lb))
    ub = prob.ub === nothing ? fill(Inf, n) : Float64.(collect(prob.ub))
    dpp = DPPData(
        n, m, c0, C, only(d0v), vec(dPm), tr.psqfns, tr.psqargs,
        conA, conb0, conB, consets,
        atomA, atomb0, atomB, [at.set for at in tr.atoms], [at.dir for at in tr.atoms],
        lb, ub, prob.sense
    )
    _warn_if_p_unused(dpp)
    return dpp
end

# Total closure capture (`f = (u, p) -> u[1] + p0[1]*u[2]` over a captured numeric
# `p0`) traces to data that does not mention `p` at all, so every re-solve would
# silently return the first answer. Partial capture is undetectable; say so.
function _warn_if_p_unused(dpp::DPPData)
    dpp.m == 0 && return nothing
    used = !iszero(dpp.C) || !iszero(dpp.dP) ||
        any(!iszero, dpp.conB) || any(!iszero, dpp.atomB)
    used && return nothing
    @warn "`p` has length $(dpp.m), but the traced objective and constraints do not " *
        "depend on it: the canonicalized problem data is constant in `p`, so " *
        "`reinit!(cache; p = …)` will return the same solution at every `p`. Did the " *
        "objective or a constraint capture a parameter vector from the enclosing scope " *
        "instead of using its own `p` argument?"
    return nothing
end

# Numeric assembly. Shared by the cold build and every `reinit!` so that bounds,
# constraints, atom cones, the objective *and its constant*, and the objective
# sense can never be refreshed one without the others. `MOI.empty!` resets the
# objective sense to FEASIBILITY_SENSE and drops the bound constraints, so both are
# re-emitted unconditionally.
function _build_moi!(model, dpp::DPPData, θ::Vector{Float64})
    n, nt = dpp.n, length(dpp.atomdirs)
    θe = _extend_theta(dpp, θ)
    c = dpp.c0 + dpp.C * θe
    d = dpp.d0 + dot(dpp.dP, θe)
    # An epigraph variable only bounds its atom on one side, so substituting it is
    # valid only where the objective pushes it against that bound: a convex atom
    # (`dir = +1`, bounded above) needs a nonnegative coefficient under MinSense, a
    # concave one (`dir = -1`, bounded below) a nonpositive one; both flip for Max.
    # The coefficient may depend on `p`, so this is re-checked numerically at every
    # `p` — it is `nt` float comparisons and does no symbolic work. Checked before
    # the model is touched, so a rejected `p` leaves the cache usable.
    sgn = dpp.sense === SciMLBase.MaxSense ? -1.0 : 1.0
    for (k, dir) in enumerate(dpp.atomdirs)
        sgn * dir * c[n + k] >= 0 || error(
            "Lowering an atom through its " * (dir > 0 ? "epigraph" : "hypograph") *
                " is valid only when the objective is " *
                (dir > 0 ? "nondecreasing" : "nonincreasing") *
                " in it for MinSense (reversed for MaxSense); got coefficient " *
                "$(c[n + k]) for $(dpp.sense)" *
                (dpp.m == 0 ? "" : " at p = $θ") * ". Route to a general " *
                "OptimizationProblem/NLP solver."
        )
    end

    MOI.empty!(model)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variables(model, n)
    t = MOI.add_variables(model, nt)
    allx = vcat(x, t)

    for i in 1:n
        dpp.lb[i] > -Inf && MOI.add_constraint(model, x[i], MOI.GreaterThan(dpp.lb[i]))
        dpp.ub[i] < Inf && MOI.add_constraint(model, x[i], MOI.LessThan(dpp.ub[i]))
    end

    # User constraints are emitted over the user columns only and in problem order,
    # so `conrefs` stays 1:1 with `prob.constraints` and therefore with `sol.dual`.
    conrefs = MOI.ConstraintIndex[]
    for k in eachindex(dpp.consets)
        b = dpp.conb0[k] + dpp.conB[k] * θe
        push!(
            conrefs,
            MOI.add_constraint(model, _affine_to_vaf(dpp.conA[k], b, x), dpp.consets[k])
        )
    end
    @assert length(conrefs) == length(dpp.consets)

    atomrefs = MOI.ConstraintIndex[]
    for j in eachindex(dpp.atomsets)
        b = dpp.atomb0[j] + dpp.atomB[j] * θe
        push!(
            atomrefs,
            MOI.add_constraint(model, _affine_to_vaf(dpp.atomA[j], b, allx), dpp.atomsets[j])
        )
    end

    # The sparsity pattern of `c` is itself θ-dependent (a coefficient that is zero
    # at one `p` is not at another), so it is recomputed here rather than cached.
    saterms = [MOI.ScalarAffineTerm(c[j], allx[j]) for j in eachindex(c) if !iszero(c[j])]
    MOI.set(
        model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(saterms, d)
    )
    MOI.set(
        model, MOI.ObjectiveSense(),
        dpp.sense === SciMLBase.MaxSense ? MOI.MAX_SENSE : MOI.MIN_SENSE
    )
    return x, conrefs, atomrefs
end

function _extend_theta(dpp::DPPData, θ::Vector{Float64})
    isempty(dpp.psqfns) && return θ
    return vcat(θ, [f(θ...) for f in dpp.psqfns])
end

# The backend's own symbolic parameters and epigraph variables. `##`-prefixed so
# they cannot compare equal to a user's own `@variables α₁` / `τ₁` — `variable` is
# keyed on the name, so a collision would silently fuse the user's symbol with ours.
const PARAM_BASE = Symbol("##p")
const TAU_BASE = Symbol("##τ")
const PSQ_BASE = Symbol("##psq")

# `u` is traced as a symbolic array rather than a vector of scalars: `norm` stays
# an inspectable atom only while its argument is an array expression.
function _symbolic_vars(prob)
    n = length(prob.u0)
    Symbolics.@variables x[1:n]
    params = if prob.p isa NullParameters
        Symbolics.Num[]   # not Float64[]: `linear_expansion(::Vector{Num}, ::Vector{Float64})`
    else                  # is a MethodError, and the extraction must run unchanged at m = 0
        prob.p isa AbstractVector || error(
            "This backend traces `p` as a vector of symbolic parameters, so a " *
                "`ConvexOptimizationProblem`'s `p` must be an `AbstractVector` or " *
                "`NullParameters`; got a `$(typeof(prob.p))`. Wrap a scalar as `[p]` and " *
                "flatten any other container."
        )
        Symbolics.Num[variable(PARAM_BASE, i) for i in eachindex(prob.p)]
    end
    return x, collect(Symbolics.scalarize(x)), params
end

"""
    AtomCone(rows, set)

Internal: one MOI cone introduced by lowering a nonlinear atom in the objective.
`rows` is the symbolic vector whose image must lie in `set`. These cones are an
implementation detail of the lowering and are deliberately kept out of
`OptimizationSolution.dual`, which stays 1:1 with the user's `ConeConstraint`s.
"""
struct AtomCone{S <: MOI.AbstractVectorSet}
    rows::Vector{Symbolics.Num}
    set::S
    dir::Int   # +1: τ bounds a convex atom above; -1: τ bounds a concave atom below
end

function _trace_problem(prob)
    vars, cols, params = _symbolic_vars(prob)
    obj = try
        prob.f.f(vars, params)
    catch e
        _trace_shape_error(e, "objective")
    end
    _check_no_unsupported_reducer(obj)
    obj = try
        _scalar(obj)
    catch e
        _trace_shape_error(e, "objective")
    end
    consvals = prob.constraints === nothing ? nothing :
        [
            try
                _asvec(con.g(vars, params))
        catch e
                _trace_shape_error(e, "constraint")
        end for con in prob.constraints
        ]
    consvals === nothing || foreach(_check_no_unsupported_reducer, consvals)
    psqvars, psqargs, psqfns = empty(params), Any[], Any[]
    if !isempty(params)
        acc = Tuple{Any, Any}[]
        optset = Set(unwrap.(cols))
        obj = Symbolics.wrap(_lift_param_squares(obj, optset, acc))
        consvals === nothing || (
            consvals = [
                Symbolics.wrap.(_lift_param_squares.(cv, Ref(optset), Ref(acc)))
                    for cv in consvals
            ]
        )
        for (s, a) in acc
            push!(psqvars, s); push!(psqargs, a)
            push!(
                psqfns, Symbolics.build_function(
                    Symbolics.wrap(unwrap(a) * unwrap(a)), params...;
                    expression = Val(false)
                )
            )
        end
    end
    objl, taus, atoms, nested = _epigraph_lower(obj)
    return (;
        vars, cols, params, psqvars, psqargs, psqfns, obj,
        objl, taus, atoms, nested, consvals,
        allcols = vcat(cols, taus),
    )
end

function _trace_shape_error(e, what)
    e isa ArgumentError &&
        occursin("different sizes", sprint(showerror, e)) || rethrow()
    return error(
        "The $what could not be traced: a product like `u' * A * B * u` " *
            "produces a `(1,)`-shaped term that cannot be combined with " *
            "scalar terms ($(sprint(showerror, e))). Write the product as " *
            "`u' * (A * B) * u` with a single matrix factor, or as the " *
            "self-product `(A * u)' * (B * u)`."
    )
end

_norm_order(t) = (a = Symbolics.arguments(t); length(a) == 1 ? 2 : Symbolics.value(a[2]))

# `norm(w, p) <= τ` is a cone membership of `(τ, w...)` for p ∈ {1, 2, Inf}; all
# three share that row layout, so only the set differs. Other `p` have no
# corresponding MOI cone.
function _norm_cone(p, dim)
    p isa Number || error(
        "The order `p` of a `norm(w, p)` atom in the objective must be a constant " *
            "1, 2 or Inf; got a non-constant expression. Route to a general " *
            "OptimizationProblem/NLP solver."
    )
    p == 2 && return MOI.SecondOrderCone(dim)
    p == 1 && return MOI.NormOneCone(dim)
    isinf(p) && p > 0 && return MOI.NormInfinityCone(dim)
    return error(
        "This backend lowers `norm(w, p)` in the objective only for p = 1, 2 or " *
            "Inf; got p = $p, which has no corresponding MathOptInterface cone. " *
            "Reformulate or route to a general OptimizationProblem/NLP solver."
    )
end

const LOWERABLE_ATOMS = (
    LinearAlgebra.norm, exp, log, abs, max, min, abs2,
    SymbolicAnalysis.quad_form,
)

function _is_lowerable_atom(ex)
    Symbolics.iscall(ex) || return false
    op = Symbolics.operation(ex)
    op isa Symbolics.SymbolicUtils.Mapreducer &&
        return _is_lowerable_reducer(op, ex) || _is_sumsq_term(op, ex)
    any(f -> op === f, LOWERABLE_ATOMS) && return true
    op === (^) && return _is_square_power(ex)
    op === (*) && return _is_quad_form_term(ex)
    return false
end

# Reductions over a symbolic array trace to a `SymbolicUtils.Mapreducer`; a
# `sum` over anything but `abs.(w)` is affine and needs no cone.
function _is_lowerable_reducer(op::SymbolicUtils.Mapreducer, ex)
    op.f === identity && op.dims isa Colon && op.init === nothing || return false
    args = Symbolics.arguments(ex)
    length(args) == 1 || return false
    op.reduce === max && return true
    op.reduce === min && return !_is_abs_broadcast(args[1])
    # `Base.add_sum` is not public API, but it is what SymbolicUtils stores as
    # the `reduce` of a traced `sum` (see `Mapreducer`'s docstring).
    return op.reduce === Base.add_sum && _is_abs_broadcast(args[1])
end

# `abs.(w)` traces to `broadcast(abs, w)`; the function is a constant symbolic.
function _is_abs_broadcast(a)
    Symbolics.iscall(a) && Symbolics.operation(a) === broadcast || return false
    bargs = Symbolics.arguments(a)
    return length(bargs) == 2 && Symbolics.value(bargs[1]) === abs
end

# `init` is silently dropped by `scalarize`/`linear_expansion`, so a `Mapreducer`
# that is not the plain scalar atom form must error here — including inside a
# lowerable atom's argument, which `_collect_atoms!` never descends into.
function _check_no_unsupported_reducer(ex)
    (ex isa Symbolics.Num || ex isa Symbolics.Arr) &&
        return _check_no_unsupported_reducer(Symbolics.unwrap(ex))
    ex isa AbstractArray && return foreach(_check_no_unsupported_reducer, ex)
    Symbolics.iscall(ex) || return nothing
    op = Symbolics.operation(ex)
    op isa SymbolicUtils.Mapreducer && _unsupported_reducer_error(op, ex)
    return foreach(_check_no_unsupported_reducer, Symbolics.arguments(ex))
end

function _unsupported_reducer_error(op::SymbolicUtils.Mapreducer, ex)
    op.dims isa Colon && op.init === nothing || error(
        "a `$(op.reduce)` reduction in the problem is supported only over all " *
            "elements (`dims` unspecified) and without `init`; got dims = " *
            "$(op.dims), init = $(op.init). Route to a general " *
            "OptimizationProblem/NLP solver."
    )
    op.f === abs && error(
        "a reduction with `abs` as the mapped function is not lowered; write " *
            "the broadcast form `sum(abs.(w))` or `maximum(abs.(w))`, or route " *
            "to a general OptimizationProblem/NLP solver."
    )
    args = Symbolics.arguments(ex)
    length(args) == 1 && op.reduce === min && _is_abs_broadcast(args[1]) &&
        error(
        "`minimum(abs.(w))` is neither convex nor concave and cannot be " *
            "lowered. Route to a general OptimizationProblem/NLP solver."
    )
    return nothing
end

# `‖w‖² <= τ` is `(τ, 1/2, w) ∈ RotatedSecondOrderCone` because `2·τ·(1/2) = τ`.

_is_square_exp(e) = (v = Symbolics.value(e); v isa Number && v == 2)
_is_array_arg(a) = Symbolics.symtype(unwrap(a)) <: AbstractArray

# `get_variables` reports the *array* symbol `x` for `norm(x)`, which is not in
# the scalarized `optset`; normalize through `scalarize` so optvars are found
# however they appear (whole array, `x[i]` getindex, or scalarized `x_i`).
function _has_optvar(ex, optset)
    return any(Symbolics.get_variables(ex)) do v
        s = Symbolics.scalarize(Symbolics.wrap(v))
        s isa AbstractArray ? any(e -> unwrap(e) in optset, vec(collect(s))) :
            unwrap(s) in optset
    end
end

function _is_square_power(ex)
    args = Symbolics.arguments(ex)
    return length(args) == 2 && _is_square_exp(args[2]) &&
        !_is_array_arg(args[1])
end

# `sum(abs2.(w))` and `sum(w .^ 2)` trace to `mapreduce(identity, add_sum,
# broadcast(f, w, …))`; `sum(abs2, w)` traces to `mapreduce(abs2, add_sum, w)`.
function _is_sumsq_term(op, ex)
    args = Symbolics.arguments(ex)
    Symbolics.symtype(unwrap(ex)) <: Number || return false
    length(args) == 1 || return false
    if op isa Symbolics.SymbolicUtils.Mapreducer{typeof(abs2), typeof(Base.add_sum)}
        return _is_array_arg(args[1])
    end
    op isa Symbolics.SymbolicUtils.Mapreducer{typeof(identity), typeof(Base.add_sum)} ||
        return false
    b = unwrap(args[1])
    Symbolics.iscall(b) && Symbolics.operation(b) === broadcast || return false
    bargs = Symbolics.arguments(b)
    f = Symbolics.value(unwrap(bargs[1]))
    f === abs2 && return length(bargs) == 2 && _is_array_arg(bargs[2])
    f === (^) || return false
    return length(bargs) == 3 && _is_square_exp(bargs[3]) && _is_array_arg(bargs[2])
end

# `v' * v` flattens to `*(adjoint(v), f…)` — `(A*u)'*(A*u)` is
# `adjoint(A*u) * A * u` — so factors after the adjoint end in `v` or multiply
# back to `v`; numeric scalars on either side fold into `P`. Returns `(v, mid)`
# (`mid` = factors between `v'` and `v`), `(v, nothing)` for a self-product.
function _quad_form_parts(ex)
    _is_scalarish(ex) || return nothing
    args = Any[unwrap(a) for a in Symbolics.arguments(ex)]
    scale = 1.0
    while !isempty(args)   # `2*u'Pu` may flatten to *(2, adjoint(u), P, u)
        s, q = _strip_scalar_mul(args[1])
        q === nothing || break
        scale *= s
        popfirst!(args)
    end
    length(args) >= 2 || return nothing
    v0 = _adjoint_arg(args[1])
    v0 === nothing && return nothing
    cL, v = _strip_scalar_mul(v0)
    v === nothing && return nothing
    scale *= cL
    rest = args[2:end]
    while !isempty(rest)
        s, q = _strip_scalar_mul(rest[end])
        q === nothing || break
        scale *= s
        pop!(rest)
    end
    isempty(rest) && return nothing
    s, vlast = _strip_scalar_mul(rest[end])
    scale *= s
    if isequal(vlast, v)
        mid = Any[rest[1:(end - 1)]...]
        scale == 1.0 || push!(mid, scale)
        return (; v, mid)
    end
    try
        isequal(_materialize_array(v), _materialize_array(foldl(*, rest))) &&
            return (; v, mid = scale == 1.0 ? nothing : Any[scale])
    catch e
        e isa InterruptException && rethrow()
    end
    return nothing
end

_is_quad_form_term(ex) = _quad_form_parts(ex) !== nothing

# `u' * A * B * u` traces to `Vector{Real}` of size `(1,)` — count that as a
# scalar term; larger arrays are not scalar objectives.
function _is_scalarish(ex)
    st = Symbolics.symtype(unwrap(ex))
    st <: Number && return true
    st <: AbstractArray || return false
    sh = Symbolics.shape(unwrap(ex))
    return isempty(sh) || prod(length, sh) == 1
end

# `(c, t)` if `t` is `*(c, …)` or a bare numeric `c`, else `(1.0, t)`;
# `p`-dependent factors are not numeric and stay put.
function _strip_scalar_mul(t)
    v = Symbolics.value(unwrap(t))
    v isa Number && return (Float64(v), nothing)
    t = unwrap(t)
    if Symbolics.iscall(t) && Symbolics.operation(t) === (*)
        a = Symbolics.arguments(t)
        v1 = Symbolics.value(unwrap(a[1]))
        v1 isa Number && length(a) >= 2 &&
            return (Float64(v1), foldl(*, Any[unwrap(x) for x in a[2:end]]))
        v2 = Symbolics.value(unwrap(a[end]))
        v2 isa Number && length(a) >= 2 &&
            return (Float64(v2), foldl(*, Any[unwrap(x) for x in a[1:(end - 1)]]))
    end
    return (1.0, t)
end

_adjoint_arg(a) = (
    a = unwrap(a);
    Symbolics.iscall(a) &&
        (Symbolics.operation(a) === adjoint || Symbolics.operation(a) === transpose) &&
        length(Symbolics.arguments(a)) == 1 || return nothing;
    unwrap(Symbolics.arguments(a)[1])
)

# Each atom becomes `(rows, set, dir)`: `rows ∈ set` ties the epigraph variable
# `tau` to the atom, and `dir` records which way it is bounded.
#   convex  `f(w) <= tau`  (epigraph,  dir = +1)
#   concave `f(w) >= tau`  (hypograph, dir = -1)
# `MOI.ExponentialCone` is {(a, b, c) : b * exp(a / b) <= c, b > 0}, so
# `exp(w) <= tau` is `(w, 1, tau)` and `log(w) >= tau` is `(tau, 1, w)`.
function _atom_lowering(t, tau)
    f = Symbolics.operation(t)
    if f === LinearAlgebra.norm
        w = _asvec(Symbolics.wrap(Symbolics.arguments(t)[1]))
        return Symbolics.Num[tau; w...], _norm_cone(_norm_order(t), length(w) + 1), 1
    end
    (f === abs2 || f === (^)) && return _rsoc_lowering(_scalar_atom_arg(t, f), tau)
    if f isa SymbolicUtils.Mapreducer
        _is_sumsq_term(f, t) && return _rsoc_lowering(_sumsq_arg(t), tau)
        return _reducer_lowering(t, f, tau)
    end
    if f === SymbolicAnalysis.quad_form
        v, P = Symbolics.arguments(t)
        return _quad_form_lowering(v, [P], tau)
    end
    if f === (*)
        parts = _quad_form_parts(t)
        parts.mid === nothing && return _rsoc_lowering(_atom_arg_vec(parts.v), tau)
        return _quad_form_lowering(parts.v, parts.mid, tau)
    end
    if f === max || f === min
        ws = _flatten_atom_args(t, f)
        rows = f === max ? Symbolics.Num[tau - Symbolics.wrap(w) for w in ws] :
            Symbolics.Num[Symbolics.wrap(w) - tau for w in ws]
        return rows, MOI.Nonnegatives(length(ws)), f === max ? 1 : -1
    end
    w = _scalar_atom_arg(t, f)
    f === abs && return Symbolics.Num[tau, w], MOI.NormOneCone(2), 1
    f === exp && return Symbolics.Num[w, 1, tau], MOI.ExponentialCone(), 1
    return Symbolics.Num[tau, 1, w], MOI.ExponentialCone(), -1
end

# `max(a, b, c)` traces as nested binary calls.
function _flatten_atom_args(t, f, ws = [])
    if Symbolics.iscall(t) && Symbolics.operation(t) === f
        for a in Symbolics.arguments(t)
            _flatten_atom_args(a, f, ws)
        end
    else
        Symbolics.symtype(t) <: Number || error(
            "`$f` in the objective is lowered only for scalar affine " *
                "arguments; got the non-scalar argument `$t`. Route to a " *
                "general OptimizationProblem/NLP solver."
        )
        push!(ws, t)
    end
    return ws
end

function _reducer_lowering(t, op::SymbolicUtils.Mapreducer, tau)
    arg = only(Symbolics.arguments(t))
    if _is_abs_broadcast(arg) && op.reduce !== min
        w = _flatvec(Symbolics.wrap(Symbolics.arguments(arg)[2]))
        cone = op.reduce === max ? MOI.NormInfinityCone(length(w) + 1) :
            MOI.NormOneCone(length(w) + 1)
        return Symbolics.Num[tau; w...], cone, 1
    end
    w = _flatvec(Symbolics.wrap(arg))
    rows = op.reduce === max ? Symbolics.Num[tau - wi for wi in w] :
        Symbolics.Num[wi - tau for wi in w]
    return rows, MOI.Nonnegatives(length(w)), op.reduce === max ? 1 : -1
end

function _flatvec(v)
    s = Symbolics.scalarize(v)
    return s isa AbstractArray ? vec(collect(s)) : [s]
end

function _scalar_atom_arg(t, f)
    a = _asvec(Symbolics.wrap(Symbolics.arguments(t)[1]))
    length(a) == 1 || error(
        "`$f` is lowered only for a scalar argument; got one of length $(length(a)). " *
            "Write the elementwise form as a sum of scalar `$f` terms, or route to a " *
            "general OptimizationProblem/NLP solver."
    )
    return only(a)
end

function _rsoc_lowering(w, tau)
    wv = w isa AbstractVector ? w : Symbolics.Num[w]
    return Symbolics.Num[tau; 1 // 2; wv...],
        MOI.RotatedSecondOrderCone(length(wv) + 2), 1
end

function _sumsq_arg(t)
    op = Symbolics.operation(t)
    arg = unwrap(Symbolics.arguments(t)[1])
    op isa Symbolics.SymbolicUtils.Mapreducer{typeof(abs2), typeof(Base.add_sum)} &&
        return _atom_arg_vec(arg)
    return _atom_arg_vec(Symbolics.arguments(arg)[2])
end

function _atom_arg_vec(v)
    s = _materialize_array(v)
    s isa AbstractVector || error(
        "The argument `$v` of a quadratic atom is not a vector expression."
    )
    return collect(s)
end

# `u' * P * u <= τ` for constant `P` is `‖L u‖² <= τ` — the same rotated-SOC
# shape as a sum of squares. `mid` is the factor(s) between `v'` and `v`:
# numeric matrices, with scalar numbers folding into `P` as a scale. `Lv` is
# also what `_reassoc_node` rewrites the form to for the DCP certificate.
function _quad_form_lowering(v, mid, tau)
    return _rsoc_lowering(_quad_form_Lv(v, mid), tau)
end

function _quad_form_Lv(v, mid)
    w = _atom_arg_vec(v)
    mid === nothing && return w
    scale = 1.0
    mats = Any[]
    for a in mid
        val = Symbolics.value(unwrap(a))
        val isa Number && (scale *= Float64(val); continue)
        push!(mats, val)
    end
    all(m -> m isa AbstractMatrix && all(x -> x isa Number, m), mats) || error(
        "`u' * P * u` / `quad_form(u, P)` is lowered only for a constant numeric " *
            "matrix `P`; got `$mid`. A `P` built from `p` moves the " *
            "cone matrix with θ and cannot be canonicalized: pass the quadratic " *
            "form differently, or use `solve(remake(prob; p = …), alg)`."
    )
    P = scale * (
        isempty(mats) ? Matrix{Float64}(I, length(w), length(w)) :
            Matrix{Float64}(foldl(*, mats))
    )
    size(P, 1) == size(P, 2) || error(
        "`u' * P * u` / `quad_form(u, P)` needs a square `P`; got size $(size(P))."
    )
    size(P, 2) == length(w) || error(
        "`u' * P * u` / `quad_form(u, P)` dimension mismatch: `P` is " *
            "$(size(P, 2))×$(size(P, 2)) but `u` has length $(length(w))."
    )
    return _psd_factor(P) * w
end

# `sym(P) = LᵀL` through the eigendecomposition: `u' * P * u ≡ u' * sym(P) * u`
# exactly. Eigenvalues in `[-n·eps·λmax, 0)` are factorization noise on a
# genuinely semidefinite `P` and are clamped — routine on rank-deficient
# inputs, so `@debug` rather than `@warn`; anything more negative is an
# indefinite form and is rejected.
function _psd_factor(P)
    F = eigen(Symmetric((P + P') / 2))
    λlo, λhi = extrema(F.values)
    λlo < -size(P, 1) * eps(Float64) * λhi && error(
        "`u' * P * u` / `quad_form(u, P)` requires a positive semidefinite `P`, " *
            "but `sym(P)` has eigenvalue $λlo: the quadratic form is not convex. " *
            "Route to a general OptimizationProblem/NLP solver."
    )
    λlo < 0 && @debug "`u' * P * u` / `quad_form(u, P)`: `sym(P)` has a small " *
        "negative eigenvalue ($λlo) within the PSD tolerance; clamping it to 0."
    return Diagonal(sqrt.(max.(F.values, 0.0))) * F.vectors'
end

function _materialize_array(s)
    sc = Symbolics.scalarize(s)
    sc isa AbstractArray && return sc
    Symbolics.symtype(sc) <: Number && return sc
    if Symbolics.iscall(sc)
        op = Symbolics.operation(sc)
        args = Symbolics.arguments(sc)
        op === Symbolics.SymbolicUtils.array_literal &&
            return reshape(collect(args[2:end]), Tuple(Symbolics.value(args[1])))
        op === (*) && return foldl(*, Any[_materialize_array(a) for a in args])
        op === adjoint && return adjoint(_materialize_array(args[1]))
        op === transpose && return transpose(_materialize_array(args[1]))
    end
    return error(
        "The expression `$s` cannot be evaluated as an array of scalar " *
            "expressions for the atom or product it appears in."
    )
end

# `c' * x` traces to a `*` term over array arguments that `scalarize` cannot
# merge into a scalar `+` (a `1×1` term added to scalar `τ` is a shape error).
function _expand_scalar_products(ex)
    ex isa Symbolics.Num && return _expand_scalar_products(unwrap(ex))
    Symbolics.iscall(ex) || return ex
    Symbolics.symtype(ex) <: AbstractArray && return ex
    op = Symbolics.operation(ex)
    args = Symbolics.arguments(ex)
    if op === (*) && any(a -> _is_array_arg(a), args)
        r = _materialize_array(ex)
        r isa AbstractArray && return length(r) == 1 ? only(r) : ex
        st = Symbolics.symtype(r)
        st <: Number && return r
        if st <: AbstractArray && _is_scalarish(r)
            sr = Symbolics.scalarize(r)
            return sr isa AbstractArray ? only(sr) : sr
        end
        return ex
    end
    newargs = map(_expand_scalar_products, args)
    all(newargs .=== args) && return ex
    return Symbolics.SymbolicUtils.maketerm(
        typeof(ex), op, newargs, Symbolics.metadata(ex)
    )
end

# `p[i]^2` / `abs2(p[i])` is a θ-dependent constant, not an epigraph atom — as
# `τ ≥ p[i]²` a `-p[i]^2` term would fail the epigraph-sign guard with a
# misleading error. Each distinct one becomes an implicit parameter column.
function _lift_param_squares(ex, optset, acc)
    ex isa Symbolics.Num && return _lift_param_squares(unwrap(ex), optset, acc)
    ex isa AbstractArray && return map(e -> _lift_param_squares(e, optset, acc), ex)
    Symbolics.iscall(ex) || return ex
    op = Symbolics.operation(ex)
    arg = if op === abs2 && !_has_optvar(ex, optset)
        Symbolics.arguments(ex)[1]
    elseif op === (^) && _is_square_power(ex) && !_has_optvar(ex, optset)
        Symbolics.arguments(ex)[1]
    end
    if arg !== nothing
        i = findfirst(t -> isequal(t[2], arg), acc)
        i === nothing || return acc[i][1]
        s = variable(PSQ_BASE, length(acc) + 1)
        push!(acc, (s, arg))
        return s
    end
    args = Symbolics.arguments(ex)
    newargs = map(a -> _lift_param_squares(a, optset, acc), args)
    all(newargs .=== args) && return ex
    return Symbolics.SymbolicUtils.maketerm(
        typeof(ex), op, newargs, Symbolics.metadata(ex)
    )
end

function _collect_atoms!(acc, ex)
    if ex isa Symbolics.Num
        return _collect_atoms!(acc, unwrap(ex))
    end
    if ex isa AbstractArray
        for e in ex
            _collect_atoms!(acc, e)
        end
        return acc
    end
    Symbolics.iscall(ex) || return acc
    if _is_lowerable_atom(ex)
        any(isequal(ex), acc) || push!(acc, ex)
        return acc
    end
    for a in Symbolics.arguments(ex)
        _collect_atoms!(acc, a)
    end
    return acc
end

function _has_op(ex, op)
    ex isa Symbolics.Num && return _has_op(unwrap(ex), op)
    ex isa AbstractArray && return any(e -> _has_op(e, op), ex)
    Symbolics.iscall(ex) || return false
    Symbolics.operation(ex) === op && return true
    return any(a -> _has_op(a, op), Symbolics.arguments(ex))
end

# A `sqrt` in the traced objective usually means a `norm` was spelled out by
# hand, so point at the spelling that keeps the atom.
_norm_hint(obj) = _has_op(obj, sqrt) ?
    " If this objective uses `norm`, spell it `norm(w)` with `w` an array " *
    "expression built from `u` (e.g. `A*u - b`, `u .- c`, `u[1:2] .- c`, or a " *
    "literal `[e1, e2, …]`); `sqrt(sum(w .^ 2))` is a different, unsupported " *
    "expression." : ""

function _collect_atoms_deep!(acc, ex)
    ex isa Symbolics.Num && return _collect_atoms_deep!(acc, unwrap(ex))
    ex isa AbstractArray &&
        return (foreach(e -> _collect_atoms_deep!(acc, e), ex); acc)
    Symbolics.iscall(ex) || return acc
    for a in Symbolics.arguments(ex)
        _collect_atoms_deep!(acc, a)
    end
    _is_lowerable_atom(ex) && (any(isequal(ex), acc) || push!(acc, ex))
    return acc
end

_has_inner_atom(t) =
    any(a -> !isempty(_collect_atoms!([], unwrap(a))), Symbolics.arguments(t))

# Inner atoms lower first and their τ substitutes into enclosing arguments
# before `scalarize` (which would rewrite `norm` into `sqrt(sum(abs2))` and
# destroy it); atoms materialized by `scalarize` are collected from the
# finished rows. `nested` routes the certificate to `_certify_nested`.
function _epigraph_lower(obj)
    queue = _collect_atoms_deep!([], unwrap(obj))
    isempty(queue) && return Symbolics.scalarize(_expand_scalar_products(unwrap(obj))),
        Symbolics.Num[], AtomCone[], false
    taus = Symbolics.Num[]
    atoms = AtomCone[]
    subs = Dict{Any, Symbolics.Num}()
    seen = Any[]
    nested = false
    while !isempty(queue)
        t = popfirst!(queue)
        any(isequal(t), seen) && continue
        push!(seen, t)
        nested |= _has_inner_atom(t)
        tsub = _subst_atom_args(t, subs)
        tau = variable(TAU_BASE, length(taus) + 1)
        rows, set, dir = _atom_lowering(tsub, tau)
        push!(taus, tau)
        push!(atoms, AtomCone(rows, set, dir))
        subs[Symbolics.wrap(t)] = tau
        for r in rows
            found = _collect_atoms!([], unwrap(r))
            isempty(found) && continue
            nested = true
            for f in found
                any(isequal(f), seen) || any(isequal(f), queue) ||
                    push!(queue, f)
            end
        end
    end
    lowered = _expand_scalar_products(Symbolics.substitute(obj, subs))
    if nested
        atoms = map(atoms) do at
            AtomCone(
                Symbolics.Num[Symbolics.substitute(r, subs) for r in at.rows],
                at.set, at.dir
            )
        end
    end
    return Symbolics.scalarize(lowered), taus, atoms, nested
end

function _subst_atom_args(t, subs)
    isempty(subs) && return t
    args = Symbolics.arguments(t)
    new = Any[
        Symbolics.unwrap(Symbolics.substitute(Symbolics.wrap(a), subs))
            for a in args
    ]
    return Symbolics.SymbolicUtils.maketerm(
        typeof(t), Symbolics.operation(t), new, Symbolics.metadata(t)
    )
end

function _asvec(v)
    s = Symbolics.scalarize(v)
    return s isa AbstractVector ? collect(s) : [s]
end
_scalar(v::AbstractVector) = only(v)
_scalar(v::AbstractArray) = error(
    "The objective of a ConvexOptimizationProblem must be a scalar; got a " *
        "$(size(v)) array. Route to a general OptimizationProblem/NLP solver."
)
_scalar(v) = v

_tofloat(x) = Float64(Symbolics.value(x))

function _affine_to_vaf(A::AbstractMatrix, b::AbstractVector, x)
    terms = MOI.VectorAffineTerm{Float64}[]
    m, n = size(A)
    for i in 1:m, j in 1:n
        iszero(A[i, j]) && continue
        push!(terms, MOI.VectorAffineTerm(i, MOI.ScalarAffineTerm(A[i, j], x[j])))
    end
    return MOI.VectorAffineFunction(terms, collect(float.(b)))
end

function _moi_status_to_retcode(s::MOI.TerminationStatusCode)
    s in (
        MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL,
        MOI.ALMOST_LOCALLY_SOLVED,
    ) && return ReturnCode.Success
    s in (
        MOI.INFEASIBLE, MOI.DUAL_INFEASIBLE, MOI.LOCALLY_INFEASIBLE,
        MOI.INFEASIBLE_OR_UNBOUNDED,
    ) && return ReturnCode.Infeasible
    s == MOI.TIME_LIMIT && return ReturnCode.MaxTime
    s in (MOI.ITERATION_LIMIT, MOI.NODE_LIMIT, MOI.SLOW_PROGRESS) &&
        return ReturnCode.MaxIters
    s in (MOI.NUMERICAL_ERROR, MOI.INVALID_MODEL, MOI.OTHER_ERROR) &&
        return ReturnCode.Failure
    return ReturnCode.Default
end

export ConvexMOI, ConeConstraint

end # module
