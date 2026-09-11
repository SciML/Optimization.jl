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

The objective may be affine or contain Euclidean-norm atoms, which are lowered
through their epigraph: `minimize norm(A*u - b, 2)` introduces an epigraph
variable `τ` with `(τ, A*u - b) ∈ SecondOrderCone` and minimizes `τ`. Keep a
`norm` argument an array expression built from `u` (`A*u - b`, `u .- c`); a
`Vector` literal of scalars scalarizes the atom away before it can be lowered.
"""
struct ConvexMOI{O} <: AbstractConvexOptAlgorithm
    optimizer_constructor::O
end

SciMLBase.allowsbounds(::AbstractConvexOptAlgorithm) = true
SciMLBase.allowsconstraints(::AbstractConvexOptAlgorithm) = true

# Must be <: AbstractOptimizationCache (build_convex_solution requires it) and
# carry real `f`/`p` fields for the solution's SymbolicIndexingInterface glue.
# No `reinit_cache` field (that would reroute getproperty(:u0/:p)).
struct ConvexOptimizationCache{F, U, P, A, AR, MOD, XV, CR, TR} <: AbstractOptimizationCache
    f::F
    u0::U
    p::P
    alg::A
    analysis::AR
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
    # One trace of `f`/`g` feeds both certification and lowering: the epigraph
    # variables minted here must be the same ones the MOI model is built from.
    tr = _trace_problem(prob)
    analysis = certify_convex(prob, tr)
    model, xvars, conrefs, atomrefs = lower_to_moi(prob, alg, tr)
    return ConvexOptimizationCache(
        prob.f, prob.u0, prob.p, alg, analysis, model, xvars, conrefs, atomrefs
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
    dual = if MOI.get(model, MOI.DualStatus()) == MOI.NO_SOLUTION
        nothing
    else
        [MOI.get(model, MOI.ConstraintDual(), c) for c in cache.conrefs]
    end
    return SciMLBase.build_convex_solution(
        cache, cache.alg, u, objective;
        dual = dual, retcode = ret, original = model,
        stats = SciMLBase.OptimizationStats()
    )
end

certify_convex(prob::ConvexOptimizationProblem) = certify_convex(prob, _trace_problem(prob))

# Certification runs on the *lowered* objective (atoms already replaced by their
# epigraph variables). Each atom's argument is proven affine separately, by
# `linear_expansion` in `lower_to_moi`, which is a stronger check than DCP.
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

function lower_to_moi(prob::ConvexOptimizationProblem, alg::ConvexMOI, tr)
    model = MOI.instantiate(alg.optimizer_constructor; with_bridge_type = Float64)
    MOI.set(model, MOI.Silent(), true)
    n = length(prob.u0)
    x = MOI.add_variables(model, n)
    t = MOI.add_variables(model, length(tr.taus))
    allx = vcat(x, t)
    cols, allcols = tr.cols, tr.allcols

    if prob.lb !== nothing
        for i in 1:n
            prob.lb[i] > -Inf &&
                MOI.add_constraint(model, x[i], MOI.GreaterThan(Float64(prob.lb[i])))
            prob.ub[i] < Inf &&
                MOI.add_constraint(model, x[i], MOI.LessThan(Float64(prob.ub[i])))
        end
    end

    # User constraints are expanded over the user columns only, so `conrefs` stays
    # 1:1 with `prob.constraints` and therefore with `sol.dual`.
    conrefs = MOI.ConstraintIndex[]
    if prob.constraints !== nothing
        for (con, gvals) in zip(prob.constraints, tr.consvals)
            A, b, islin = linear_expansion(gvals, cols)   # gvals == A*cols + b
            islin || error("Constraint $(con.set) is not affine in the variables.")
            f = _affine_to_vaf(_tofloat.(A), _tofloat.(b), x)
            push!(conrefs, MOI.add_constraint(model, f, con.set))
        end
    end
    @assert length(conrefs) ==
        (prob.constraints === nothing ? 0 : length(prob.constraints))

    atomrefs = MOI.ConstraintIndex[]
    for at in tr.atoms
        A, b, islin = linear_expansion(at.rows, allcols)
        islin || error(
            "The argument of a `norm` atom in the objective must be affine in the " *
                "optimization variables."
        )
        f = _affine_to_vaf(_tofloat.(A), _tofloat.(b), allx)
        push!(atomrefs, MOI.add_constraint(model, f, at.set))
    end

    Ao, bo, olin = linear_expansion(_asvec(tr.objl), allcols)
    olin || error(
        "This backend requires an objective that is affine in the optimization " *
            "variables and in the epigraph variables of its lowered atoms " *
            "(only `norm(w, 2)` is lowered so far)."
    )
    c = vec(_tofloat.(Ao))
    d = _tofloat(only(bo))
    # `norm(w) <= τ` only bounds the atom from above, so replacing the atom by τ is
    # valid only where the objective is nondecreasing in τ (nonincreasing for Max).
    sgn = prob.sense === SciMLBase.MaxSense ? -1.0 : 1.0
    for k in eachindex(tr.taus)
        sgn * c[n + k] >= 0 || error(
            "Epigraph lowering of a `norm` atom is valid only when the objective is " *
                "nondecreasing in it for MinSense (nonincreasing for MaxSense); got " *
                "coefficient $(c[n + k]) for $(prob.sense). Route to a general " *
                "OptimizationProblem/NLP solver."
        )
    end
    saterms = [MOI.ScalarAffineTerm(c[j], allx[j]) for j in eachindex(c) if !iszero(c[j])]
    MOI.set(
        model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(saterms, d)
    )
    MOI.set(
        model, MOI.ObjectiveSense(),
        prob.sense === SciMLBase.MaxSense ? MOI.MAX_SENSE : MOI.MIN_SENSE
    )
    return model, x, conrefs, atomrefs
end

# `u` is traced as a symbolic array rather than a vector of scalars: `norm` stays
# an inspectable atom only while its argument is an array expression.
function _symbolic_vars(prob)
    n = length(prob.u0)
    Symbolics.@variables x[1:n]
    params = prob.p isa NullParameters ? Float64[] :
        [variable(:α, i) for i in eachindex(prob.p)]
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

function _is_lowerable_atom(ex)
    Symbolics.iscall(ex) || return false
    return Symbolics.operation(ex) === LinearAlgebra.norm
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
        p = _norm_order(t)
        (p isa Number && p == 2) || error(
            "This backend lowers only the Euclidean norm `norm(w)` / `norm(w, 2)` " *
                "in the objective; got `norm(w, p)` with p = " *
                (p isa Number ? string(p) : "a non-constant expression") *
                ", which is not a second-order cone. Reformulate or route to a " *
                "general OptimizationProblem/NLP solver."
        )
        w = _asvec(Symbolics.wrap(Symbolics.arguments(t)[1]))
        tau = variable(:τ, k)
        push!(taus, tau)
        push!(atoms, AtomCone(Symbolics.Num[tau; w...], MOI.SecondOrderCone(length(w) + 1)))
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
