module OptimizationMOI

using Reexport
@reexport using Optimization
using MathOptInterface
using Optimization.SciMLBase
using SciMLStructures
using SymbolicIndexingInterface
using SparseArrays
import ModelingToolkit: parameters, unknowns, varmap_to_vars, mergedefaults, toexpr
import ModelingToolkit
const MTK = ModelingToolkit
using Symbolics
using LinearAlgebra

const MOI = MathOptInterface

const DenseOrSparse{T} = Union{Matrix{T}, SparseMatrixCSC{T}}

function SciMLBase.requiresgradient(opt::Union{
        MOI.AbstractOptimizer, MOI.OptimizerWithAttributes})
    true
end
function SciMLBase.requireshessian(opt::Union{
        MOI.AbstractOptimizer, MOI.OptimizerWithAttributes})
    true
end
function SciMLBase.requiresconsjac(opt::Union{
        MOI.AbstractOptimizer, MOI.OptimizerWithAttributes})
    true
end
function SciMLBase.requiresconshess(opt::Union{
        MOI.AbstractOptimizer, MOI.OptimizerWithAttributes})
    true
end

function SciMLBase.allowsbounds(opt::Union{MOI.AbstractOptimizer,
        MOI.OptimizerWithAttributes})
    true
end
function SciMLBase.allowsconstraints(opt::Union{MOI.AbstractOptimizer,
        MOI.OptimizerWithAttributes})
    true
end

function _create_new_optimizer(opt::MOI.OptimizerWithAttributes)
    return _create_new_optimizer(MOI.instantiate(opt, with_bridge_type = Float64))
end

function _create_new_optimizer(opt::MOI.AbstractOptimizer)
    if !MOI.is_empty(opt)
        MOI.empty!(opt) # important! ensure that the optimizer is empty
    end
    if MOI.supports_incremental_interface(opt)
        return opt
    end
    opt_setup = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{
            Float64,
        }()),
        opt)
    return opt_setup
end

function __map_optimizer_args(cache,
        opt::Union{MOI.AbstractOptimizer, MOI.OptimizerWithAttributes
        };
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        kwargs...)
    optimizer = _create_new_optimizer(opt)
    for (key, value) in kwargs
        MOI.set(optimizer, MOI.RawOptimizerAttribute("$(key)"), value)
    end
    if !isnothing(maxtime)
        MOI.set(optimizer, MOI.TimeLimitSec(), maxtime)
    end
    if !isnothing(reltol)
        @warn "common reltol argument is currently not used by $(optimizer). Set tolerances via optimizer specific keyword arguments."
    end
    if !isnothing(abstol)
        @warn "common abstol argument is currently not used by $(optimizer). Set tolerances via optimizer specific keyword arguments."
    end
    if !isnothing(maxiters)
        @warn "common maxiters argument is currently not used by $(optimizer). Set number of iterations via optimizer specific keyword arguments."
    end
    return optimizer
end

function __moi_status_to_ReturnCode(status::MOI.TerminationStatusCode)
    if status in [
        MOI.OPTIMAL,
        MOI.LOCALLY_SOLVED,
        MOI.ALMOST_OPTIMAL,
        MOI.ALMOST_LOCALLY_SOLVED
    ]
        return ReturnCode.Success
    elseif status in [
        MOI.INFEASIBLE,
        MOI.DUAL_INFEASIBLE,
        MOI.LOCALLY_INFEASIBLE,
        MOI.INFEASIBLE_OR_UNBOUNDED,
        MOI.ALMOST_INFEASIBLE,
        MOI.ALMOST_DUAL_INFEASIBLE
    ]
        return ReturnCode.Infeasible
    elseif status in [
        MOI.ITERATION_LIMIT,
        MOI.NODE_LIMIT,
        MOI.SLOW_PROGRESS
    ]
        return ReturnCode.MaxIters
    elseif status == MOI.TIME_LIMIT
        return ReturnCode.MaxTime
    elseif status in [
        MOI.OPTIMIZE_NOT_CALLED,
        MOI.NUMERICAL_ERROR,
        MOI.INVALID_MODEL,
        MOI.INVALID_OPTION,
        MOI.INTERRUPTED,
        MOI.OTHER_ERROR,
        MOI.SOLUTION_LIMIT,
        MOI.MEMORY_LIMIT,
        MOI.OBJECTIVE_LIMIT,
        MOI.NORM_LIMIT,
        MOI.OTHER_LIMIT
    ]
        return ReturnCode.Failure
    else
        return ReturnCode.Default
    end
end

_get_variable_index_from_expr(expr::T) where {T} = throw(MalformedExprException("$expr"))
function _get_variable_index_from_expr(expr::Expr)
    _is_var_ref_expr(expr)
    return MOI.VariableIndex(expr.args[2])
end

function _is_var_ref_expr(expr::Expr)
    expr.head == :ref || throw(MalformedExprException("$expr")) # x[i]
    expr.args[1] == :x || throw(MalformedExprException("$expr"))
    return true
end

function is_eq(expr::Expr)
    expr.head == :call || throw(MalformedExprException("$expr"))
    expr.args[1] in [:(==), :(=)]
end

function is_leq(expr::Expr)
    expr.head == :call || throw(MalformedExprException("$expr"))
    expr.args[1] == :(<=)
end

"""
    rep_pars_vals!(expr::T, expr_map)

Replaces variable expressions of the form `:some_variable` or `:(getindex, :some_variable, j)` with
`x[i]` were `i` is the corresponding index in the state vector. Same for the parameters. The
variable/parameter pairs are provided via the `expr_map`.

Expects only expressions where the variables and parameters are of the form `:some_variable`
or `:(getindex, :some_variable, j)` or :(some_variable[j]).
"""
rep_pars_vals!(expr::T, expr_map) where {T} = expr
function rep_pars_vals!(expr::Symbol, expr_map)
    for (f, n) in expr_map
        isequal(f, expr) && return n
    end
    return expr
end
function rep_pars_vals!(expr::Expr, expr_map)
    if (expr.head == :call && expr.args[1] == getindex) || (expr.head == :ref)
        for (f, n) in expr_map
            isequal(f, expr) && return n
        end
    end
    Threads.@sync for i in eachindex(expr.args)
        i == 1 && expr.head == :call && continue # first arg is the operator
        Threads.@spawn expr.args[i] = rep_pars_vals!(expr.args[i], expr_map)
    end
    return expr
end

"""
    symbolify!(e)

Ensures that a given expression is fully symbolic, e.g. no function calls.
"""
symbolify!(e) = e
function symbolify!(e::Expr)
    if !(e.args[1] isa Symbol)
        e.args[1] = Symbol(e.args[1])
    end
    symbolify!.(e.args)
    return e
end

"""
    convert_to_expr(eq, sys; expand_expr = false, pairs_arr = expr_map(sys))

Converts the given symbolic expression to a Julia `Expr` and replaces all symbols, i.e. unknowns and
parameters with `x[i]` and `p[i]`.

# Arguments:

  - `eq`: Expression to convert
  - `sys`: Reference to the system holding the parameters and unknowns
  - `expand_expr=false`: If `true` the symbolic expression is expanded first.
"""
function convert_to_expr(eq, expr_map; expand_expr = false)
    if expand_expr
        eq = try
            Symbolics.expand(eq) # PolyForm sometimes errors
        catch e
            Symbolics.expand(eq)
        end
    end
    expr = ModelingToolkit.toexpr(eq)

    expr = rep_pars_vals!(expr, expr_map)
    expr = symbolify!(expr)
    return expr
end

function get_expr_map(sys)
    dvs = ModelingToolkit.unknowns(sys)
    ps = ModelingToolkit.parameters(sys)
    return vcat(
        [ModelingToolkit.toexpr(_s) => Expr(:ref, :x, i)
         for (i, _s) in enumerate(dvs)],
        [ModelingToolkit.toexpr(_p) => Expr(:ref, :p, i)
         for (i, _p) in enumerate(ps)])
end

"""
Replaces every expression `:x[i]` with `:x[MOI.VariableIndex(i)]`
"""
_replace_variable_indices!(expr) = expr
function _replace_variable_indices!(expr::Expr)
    if expr.head == :ref && expr.args[1] == :x
        return Expr(:ref, :x, MOI.VariableIndex(expr.args[2]))
    end
    for i in 1:length(expr.args)
        expr.args[i] = _replace_variable_indices!(expr.args[i])
    end
    return expr
end

"""
Replaces every expression `:p[i]` with its numeric value from `p`
"""
_replace_parameter_indices!(expr, p) = expr
function _replace_parameter_indices!(expr::Expr, p)
    if expr.head == :ref && expr.args[1] == :p
        tunable, _, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), p)
        p_ = tunable[expr.args[2]]
        (!isa(p_, Real) || isnan(p_) || isinf(p_)) &&
            throw(ArgumentError("Expected parameters to be real valued: $(expr.args[2]) => $p_"))
        return p_
    end
    for i in 1:length(expr.args)
        expr.args[i] = _replace_parameter_indices!(expr.args[i], p)
    end
    return expr
end

"""
Replaces calls like `:(getindex, 1, :x)` with `:(x[1])`
"""
repl_getindex!(expr::T) where {T} = expr
function repl_getindex!(expr::Expr)
    if expr.head == :call && expr.args[1] == :getindex
        return Expr(:ref, expr.args[2], expr.args[3])
    end
    for i in 1:length(expr.args)
        expr.args[i] = repl_getindex!(expr.args[i])
    end
    return expr
end

include("nlp.jl")
include("moi.jl")

function SciMLBase.supports_opt_cache_interface(alg::Union{MOI.AbstractOptimizer,
        MOI.OptimizerWithAttributes})
    true
end

function SciMLBase.__init(prob::OptimizationProblem,
        opt::Union{MOI.AbstractOptimizer, MOI.OptimizerWithAttributes};
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        mtkize = false,
        kwargs...)
    cache = if MOI.supports(_create_new_optimizer(opt), MOI.NLPBlock())
        MOIOptimizationNLPCache(prob,
            opt;
            maxiters,
            maxtime,
            abstol,
            reltol,
            mtkize,
            kwargs...)
    else
        MOIOptimizationCache(prob, opt; maxiters, maxtime, abstol, reltol, kwargs...)
    end
    return cache
end

end
