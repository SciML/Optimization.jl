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

Conic backend: certify convexity with SymbolicAnalysis, lower the (affine)
objective and each `ConeConstraint` to a MathOptInterface cone, and solve with
`optimizer_constructor`.
"""
struct ConvexMOI{O} <: AbstractConvexOptAlgorithm
    optimizer_constructor::O
end

SciMLBase.allowsbounds(::AbstractConvexOptAlgorithm) = true
SciMLBase.allowsconstraints(::AbstractConvexOptAlgorithm) = true

# Must be <: AbstractOptimizationCache (build_convex_solution requires it) and
# carry real `f`/`p` fields for the solution's SymbolicIndexingInterface glue.
# No `reinit_cache` field (that would reroute getproperty(:u0/:p)).
struct ConvexOptimizationCache{F, U, P, A, AR, MOD, XV, CR} <: AbstractOptimizationCache
    f::F
    u0::U
    p::P
    alg::A
    analysis::AR
    model::MOD           # lowered MOI model
    xvars::XV            # Vector{MOI.VariableIndex}: user u -> MOI variables
    conrefs::CR          # Vector{MOI.ConstraintIndex}, 1:1 with prob.constraints
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
    analysis = certify_convex(prob)
    model, xvars, conrefs = lower_to_moi(prob, alg)
    return ConvexOptimizationCache(
        prob.f, prob.u0, prob.p, alg, analysis, model, xvars, conrefs
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

function certify_convex(prob::ConvexOptimizationProblem)
    vars, params = _symbolic_vars(prob)
    obj = unwrap(_scalar(prob.f.f(vars, params)))
    obj_res = analyze(obj)
    ok = prob.sense === SciMLBase.MaxSense ?
        obj_res.curvature in (SymbolicAnalysis.Concave, SymbolicAnalysis.Affine) :
        obj_res.curvature in (SymbolicAnalysis.Convex, SymbolicAnalysis.Affine)
    ok || error(
        "Objective is not certified convex for $(prob.sense): curvature = " *
            "$(obj_res.curvature). Route to a general OptimizationProblem/NLP solver."
    )
    cons_res = _certify_constraints(prob, vars, params)
    return (; objective = obj_res, constraints = cons_res)
end

# MVP: constraints are affine-in-cone, so every output component must be Affine.
function _certify_constraints(prob, vars, params)
    prob.constraints === nothing && return nothing
    res = []
    for con in prob.constraints
        cres = analyze.(unwrap.(_asvec(con.g(vars, params))))
        all(r -> r.curvature == SymbolicAnalysis.Affine, cres) || error(
            "This backend supports affine-in-cone constraints only; got " *
                "curvatures $(getproperty.(cres, :curvature)) for cone $(con.set)."
        )
        push!(res, cres)
    end
    return res
end

function lower_to_moi(prob::ConvexOptimizationProblem, alg::ConvexMOI)
    model = MOI.instantiate(alg.optimizer_constructor; with_bridge_type = Float64)
    MOI.set(model, MOI.Silent(), true)
    n = length(prob.u0)
    x = MOI.add_variables(model, n)
    vars, params = _symbolic_vars(prob)

    if prob.lb !== nothing
        for i in 1:n
            prob.lb[i] > -Inf &&
                MOI.add_constraint(model, x[i], MOI.GreaterThan(Float64(prob.lb[i])))
            prob.ub[i] < Inf &&
                MOI.add_constraint(model, x[i], MOI.LessThan(Float64(prob.ub[i])))
        end
    end

    conrefs = MOI.ConstraintIndex[]
    if prob.constraints !== nothing
        for con in prob.constraints
            gvals = _asvec(con.g(vars, params))
            A, b, islin = linear_expansion(gvals, vars)   # gvals == A*vars + b
            islin || error("Constraint $(con.set) is not affine in the variables.")
            f = _affine_to_vaf(_tofloat.(A), _tofloat.(b), x)
            push!(conrefs, MOI.add_constraint(model, f, con.set))
        end
    end

    objexpr = _asvec(_scalar(prob.f.f(vars, params)))
    Ao, bo, olin = linear_expansion(objexpr, vars)
    olin || error("This backend requires an affine objective.")
    c = vec(_tofloat.(Ao))
    d = _tofloat(only(bo))
    saterms = [MOI.ScalarAffineTerm(c[j], x[j]) for j in 1:n if !iszero(c[j])]
    MOI.set(
        model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(saterms, d)
    )
    MOI.set(
        model, MOI.ObjectiveSense(),
        prob.sense === SciMLBase.MaxSense ? MOI.MAX_SENSE : MOI.MIN_SENSE
    )
    return model, x, conrefs
end

function _symbolic_vars(prob)
    vars = [variable(:x, i) for i in 1:length(prob.u0)]
    params = prob.p isa NullParameters ? Float64[] :
        [variable(:α, i) for i in eachindex(prob.p)]
    return vars, params
end

_asvec(v::AbstractVector) = v
_asvec(v) = [v]
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
