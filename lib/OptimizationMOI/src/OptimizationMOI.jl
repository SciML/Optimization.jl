module OptimizationMOI

using Reexport
@reexport using Optimization
using MathOptInterface
using Optimization.SciMLBase
using SparseArrays
import ModelingToolkit
using ModelingToolkit: parameters, states, varmap_to_vars, mergedefaults
const MTK = ModelingToolkit
import Symbolics

const MOI = MathOptInterface

const DenseOrSparse{T} = Union{Matrix{T}, SparseMatrixCSC{T}}

function SciMLBase.allowsbounds(opt::Union{MOI.AbstractOptimizer,
    MOI.OptimizerWithAttributes})
    true
end
function SciMLBase.allowsconstraints(opt::Union{MOI.AbstractOptimizer,
    MOI.OptimizerWithAttributes})
    true
end
function SciMLBase.allowscallback(opt::Union{MOI.AbstractOptimizer,
    MOI.OptimizerWithAttributes})
    false
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
    opt_setup = MOI.Utilities.CachingOptimizer(MOI.Utilities.UniversalFallback(MOI.Utilities.Model{
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
        MOI.ALMOST_LOCALLY_SOLVED,
    ]
        return ReturnCode.Success
    elseif status in [
        MOI.INFEASIBLE,
        MOI.DUAL_INFEASIBLE,
        MOI.LOCALLY_INFEASIBLE,
        MOI.INFEASIBLE_OR_UNBOUNDED,
        MOI.ALMOST_INFEASIBLE,
        MOI.ALMOST_DUAL_INFEASIBLE,
    ]
        return ReturnCode.Infeasible
    elseif status in [
        MOI.ITERATION_LIMIT,
        MOI.NODE_LIMIT,
        MOI.SLOW_PROGRESS,
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
        MOI.OTHER_LIMIT,
    ]
        return ReturnCode.Failure
    else
        return ReturnCode.Default
    end
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
        p_ = p[expr.args[2]]
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
    kwargs...)
    cache = if MOI.supports(_create_new_optimizer(opt), MOI.NLPBlock())
        MOIOptimizationNLPCache(prob, opt; maxiters, maxtime, abstol, reltol, kwargs...)
    else
        MOIOptimizationCache(prob, opt; maxiters, maxtime, abstol, reltol, kwargs...)
    end
    return cache
end

end
