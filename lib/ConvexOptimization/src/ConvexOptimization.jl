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
`(τ, A*u - b) ∈ SecondOrderCone` and minimizes `τ`. Supported atoms are

  - `norm(w, p)` for `p = 1, 2, Inf` (`NormOneCone`, `SecondOrderCone`,
    `NormInfinityCone`), with `w` an array expression in `u`;
  - `exp(w)` and `log(w)` for scalar affine `w` (`ExponentialCone`); `log` is
    concave, so it is bounded below and must enter the objective negatively
    (e.g. a `-log` barrier).

Keep a `norm` argument an array expression built from `u` (`A*u - b`, `u .- c`);
a `Vector` literal of scalars scalarizes the atom away before it can be lowered.

Parameters are first-class: see [`SciMLBase.reinit!`](@ref) for re-solving at a new
`p` without re-running the symbolic canonicalization.
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
    # With symbolic parameters `analyze` sees `p[1]*u[2]` as a product of two
    # non-constant symbols and reports UnknownCurvature, and substituting a numeric
    # `p` first would certify one `p` only. On the parametric path the structural
    # predicate in `_dpp_extract` is the certificate instead: it is strictly stronger
    # than DCP (the lowered problem *is* a cone program with fixed cones).
    analysis = isempty(tr.params) ? certify_convex(prob, tr) : nothing
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
function certify_convex(prob::ConvexOptimizationProblem, tr)
    obj_res = analyze(unwrap(tr.objl))
    ok = prob.sense === SciMLBase.MaxSense ?
        obj_res.curvature in (SymbolicAnalysis.Concave, SymbolicAnalysis.Affine) :
        obj_res.curvature in (SymbolicAnalysis.Convex, SymbolicAnalysis.Affine)
    ok || error(
        "Objective is not certified convex for $(prob.sense): curvature = " *
            "$(obj_res.curvature). Route to a general OptimizationProblem/NLP solver." *
            _norm_hint(tr.obj)
    )
    cons_res = _certify_constraints(prob, tr)
    return (; objective = obj_res, constraints = cons_res)
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
"""
struct DPPData{CS, AS, SE}
    n::Int                          # user variables
    m::Int                          # parameters
    c0::Vector{Float64}             # length n + ntau
    C::Matrix{Float64}              # (n + ntau) × m
    d0::Float64
    dP::Vector{Float64}             # length m
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
    params, cols, allcols = tr.params, tr.cols, tr.allcols
    paramset = Set(unwrap.(params))
    tauset = Set(unwrap.(tr.taus))
    m = length(params)

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
            "The argument of a `norm` atom in the objective must be affine in the " *
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
        n, m, c0, C, only(d0v), vec(dPm),
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
    c = dpp.c0 + dpp.C * θ
    d = dpp.d0 + dot(dpp.dP, θ)
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
        b = dpp.conb0[k] + dpp.conB[k] * θ
        push!(
            conrefs,
            MOI.add_constraint(model, _affine_to_vaf(dpp.conA[k], b, x), dpp.consets[k])
        )
    end
    @assert length(conrefs) == length(dpp.consets)

    atomrefs = MOI.ConstraintIndex[]
    for j in eachindex(dpp.atomsets)
        b = dpp.atomb0[j] + dpp.atomB[j] * θ
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

# The backend's own symbolic parameters and epigraph variables. `##`-prefixed so
# they cannot compare equal to a user's own `@variables α₁` / `τ₁` — `variable` is
# keyed on the name, so a collision would silently fuse the user's symbol with ours.
const PARAM_BASE = Symbol("##p")
const TAU_BASE = Symbol("##τ")

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
        [variable(PARAM_BASE, i) for i in eachindex(prob.p)]
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
    obj = _scalar(prob.f.f(vars, params))
    objl, taus, atoms = _epigraph_lower(obj)
    consvals = prob.constraints === nothing ? nothing :
        [_asvec(con.g(vars, params)) for con in prob.constraints]
    return (;
        vars, cols, params, obj,
        objl, taus, atoms, consvals,
        allcols = vcat(cols, taus),
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

const LOWERABLE_ATOMS = (LinearAlgebra.norm, exp, log)

function _is_lowerable_atom(ex)
    Symbolics.iscall(ex) || return false
    return any(f -> Symbolics.operation(ex) === f, LOWERABLE_ATOMS)
end

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
    w = _scalar_atom_arg(t, f)
    f === exp && return Symbolics.Num[w, 1, tau], MOI.ExponentialCone(), 1
    return Symbolics.Num[tau, 1, w], MOI.ExponentialCone(), -1
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

# A `sqrt` in the traced objective usually means a `norm` was scalarized away
# before we could see it, so point at the spelling that preserves the atom.
_norm_hint(obj) = _has_op(obj, sqrt) ?
    " If this objective uses `norm`, keep its argument an array expression built " *
    "from `u` (e.g. `A*u - b`, `u .- c`, `u[1:2] .- c`); a `Vector` literal such as " *
    "`[u[1]-1, u[2]-2]` destroys the `norm` atom and cannot be lowered." : ""

# Replace each `norm(w, 2)` in the objective by a fresh epigraph variable τ and
# record `(τ, w...) ∈ SecondOrderCone`. `scalarize` afterwards so the residual
# objective is analyzed with the same scalar semantics as a non-atom objective.
function _epigraph_lower(obj)
    nodes = _collect_atoms!([], unwrap(obj))
    isempty(nodes) && return Symbolics.scalarize(obj), Symbolics.Num[], AtomCone[]
    taus = Symbolics.Num[]
    atoms = AtomCone[]
    subs = Dict{Any, Symbolics.Num}()
    for (k, t) in enumerate(nodes)
        tau = variable(TAU_BASE, k)
        rows, set, dir = _atom_lowering(t, tau)
        push!(taus, tau)
        push!(atoms, AtomCone(rows, set, dir))
        subs[Symbolics.wrap(t)] = tau
    end
    return Symbolics.scalarize(Symbolics.substitute(obj, subs)), taus, atoms
end

function _asvec(v)
    s = Symbolics.scalarize(v)
    return s isa AbstractVector ? collect(s) : [s]
end
_scalar(v::AbstractVector) = only(v)
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
