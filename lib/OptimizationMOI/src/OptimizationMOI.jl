module OptimizationMOI

using Reexport
@reexport using Optimization
using MathOptInterface, Optimization.SciMLBase, SparseArrays

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

struct MOIOptimizationProblem{T, F <: OptimizationFunction, uType, P,
                              JT <: DenseOrSparse{T}, HT <: DenseOrSparse{T},
                              CHT <: DenseOrSparse{T}} <:
       MOI.AbstractNLPEvaluator
    f::F
    u0::uType
    p::P
    J::JT
    H::HT
    cons_H::Vector{CHT}
    lcons::Vector{T}
    ucons::Vector{T}
end

function MOIOptimizationProblem(prob::OptimizationProblem)
    num_cons = prob.ucons === nothing ? 0 : length(prob.ucons)
    f = Optimization.instantiate_function(prob.f, prob.u0, prob.f.adtype, prob.p, num_cons)
    T = eltype(prob.u0)
    n = length(prob.u0)
    return MOIOptimizationProblem(f,
                                  prob.u0,
                                  prob.p,
                                  isnothing(f.cons_jac_prototype) ? zeros(T, num_cons, n) :
                                  convert.(T, f.cons_jac_prototype),
                                  isnothing(f.hess_prototype) ? zeros(T, n, n) :
                                  convert.(T, f.hess_prototype),
                                  isnothing(f.cons_hess_prototype) ?
                                  Matrix{T}[zeros(T, n, n) for i in 1:num_cons] :
                                  [convert.(T, f.cons_hess_prototype[i])
                                   for i in 1:num_cons],
                                  prob.lcons === nothing ? fill(-Inf, num_cons) :
                                  prob.lcons,
                                  prob.ucons === nothing ? fill(Inf, num_cons) : prob.ucons)
end

function MOI.features_available(prob::MOIOptimizationProblem)
    features = [:Grad, :Hess, :Jac]
    # Assume that if there are constraints and expr then cons_expr exists
    if prob.f.expr !== nothing
        push!(features, :ExprGraph)
    end
    return features
end

function MOI.initialize(moiproblem::MOIOptimizationProblem,
                        requested_features::Vector{Symbol})
    available_features = MOI.features_available(moiproblem)
    for feat in requested_features
        if !(feat in available_features)
            error("Unsupported feature $feat")
            # TODO: implement Jac-vec and Hess-vec products
            # for solvers that need them
        end
    end
    return
end

function MOI.eval_objective(moiproblem::MOIOptimizationProblem, x)
    return moiproblem.f(x, moiproblem.p)
end

function MOI.eval_constraint(moiproblem::MOIOptimizationProblem, g, x)
    moiproblem.f.cons(g, x)
    return
end

function MOI.eval_objective_gradient(moiproblem::MOIOptimizationProblem, G, x)
    moiproblem.f.grad(G, x)
    return
end

# This structure assumes the calculation of moiproblem.J is dense.
function MOI.jacobian_structure(moiproblem::MOIOptimizationProblem)
    if moiproblem.J isa SparseMatrixCSC
        rows, cols, _ = findnz(moiproblem.J)
        inds = Tuple{Int, Int}[(i, j) for (i, j) in zip(rows, cols)]
    else
        rows, cols = size(moiproblem.J)
        inds = Tuple{Int, Int}[(i, j) for j in 1:cols for i in 1:rows]
    end
    return inds
end

function MOI.eval_constraint_jacobian(moiproblem::MOIOptimizationProblem, j, x)
    if isempty(j)
        return
    elseif moiproblem.f.cons_j === nothing
        error("Use OptimizationFunction to pass the derivatives or " *
              "automatically generate them with one of the autodiff backends")
    end
    moiproblem.f.cons_j(moiproblem.J, x)
    if moiproblem.J isa SparseMatrixCSC
        nnz = nonzeros(moiproblem.J)
        @assert length(j) == length(nnz)
        for (i, Ji) in zip(eachindex(j), nnz)
            j[i] = Ji
        end
    else
        for i in eachindex(j)
            j[i] = moiproblem.J[i]
        end
    end
    return
end

function MOI.hessian_lagrangian_structure(moiproblem::MOIOptimizationProblem)
    sparse_obj = moiproblem.H isa SparseMatrixCSC
    sparse_constraints = all(H -> H isa SparseMatrixCSC, moiproblem.cons_H)
    if !sparse_constraints && any(H -> H isa SparseMatrixCSC, moiproblem.cons_H)
        # Some constraint hessians are dense and some are sparse! :(
        error("Mix of sparse and dense constraint hessians are not supported")
    end
    N = length(moiproblem.u0)
    inds = if sparse_obj
        rows, cols, _ = findnz(moiproblem.H)
        Tuple{Int, Int}[(i, j) for (i, j) in zip(rows, cols) if i <= j]
    else
        Tuple{Int, Int}[(row, col) for col in 1:N for row in 1:col]
    end
    if sparse_constraints
        for Hi in moiproblem.cons_H
            r, c, _ = findnz(Hi)
            for (i, j) in zip(r, c)
                if i <= j
                    push!(inds, (i, j))
                end
            end
        end
    elseif !sparse_obj
        # Performance optimization. If both are dense, no need to repeat
    else
        for col in 1:N, row in 1:col
            push!(inds, (row, col))
        end
    end
    return inds
end

function MOI.eval_hessian_lagrangian(moiproblem::MOIOptimizationProblem{T},
                                     h,
                                     x,
                                     σ,
                                     μ) where {T}
    fill!(h, zero(T))
    k = 0
    moiproblem.f.hess(moiproblem.H, x)
    sparse_objective = moiproblem.H isa SparseMatrixCSC
    if sparse_objective
        rows, cols, _ = findnz(moiproblem.H)
        for (i, j) in zip(rows, cols)
            if i <= j
                k += 1
                h[k] = σ * moiproblem.H[i, j]
            end
        end
    else
        for i in 1:size(moiproblem.H, 1), j in 1:i
            k += 1
            h[k] = σ * moiproblem.H[i, j]
        end
    end
    # A count of the number of non-zeros in the objective Hessian is needed if
    # the constraints are dense.
    nnz_objective = k
    if !isempty(μ) && !all(iszero, μ)
        moiproblem.f.cons_h(moiproblem.cons_H, x)
        for (μi, Hi) in zip(μ, moiproblem.cons_H)
            if Hi isa SparseMatrixCSC
                rows, cols, _ = findnz(Hi)
                for (i, j) in zip(rows, cols)
                    if i <= j
                        k += 1
                        h[k] += μi * Hi[i, j]
                    end
                end
            else
                # The constraints are dense. We only store one copy of the
                # Hessian, so reset `k` to where it starts. That will be
                # `nnz_objective` if the objective is sprase, and `0` otherwise.
                k = sparse_objective ? nnz_objective : 0
                for i in 1:size(Hi, 1), j in 1:i
                    k += 1
                    h[k] += μi * Hi[i, j]
                end
            end
        end
    end
    return
end

_replace_variable_indices(expr) = expr

function _replace_variable_indices(expr::Expr)
    if expr.head == :ref
        @assert length(expr.args) == 2
        @assert expr.args[1] == :x
        return Expr(:ref, :x, MOI.VariableIndex(expr.args[2]))
    end
    for i in 1:length(expr.args)
        expr.args[i] = _replace_variable_indices(expr.args[i])
    end
    return expr
end

function MOI.objective_expr(prob::MOIOptimizationProblem)
    return _replace_variable_indices(prob.f.expr)
end

function MOI.constraint_expr(prob::MOIOptimizationProblem, i)
    # expr has the form f(x) == 0
    expr = _replace_variable_indices(prob.f.cons_expr[i].args[2])
    lb, ub = prob.lcons[i], prob.ucons[i]
    return :($lb <= $expr <= $ub)
end

function _create_new_optimizer(opt::MOI.OptimizerWithAttributes)
    return _create_new_optimizer(MOI.instantiate(opt))
end

function _create_new_optimizer(model::MOI.AbstractOptimizer)
    if !MOI.is_empty(model)
        MOI.empty!(model)
    end
    if MOI.supports_incremental_interface(model)
        return model
    end
    return MOI.Utilities.CachingOptimizer(MOI.Utilities.UniversalFallback(MOI.Utilities.Model{
                                                                                              Float64
                                                                                              }()),
                                          model)
end

function __map_optimizer_args(prob::OptimizationProblem,
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

function _create_integer_domain(id::Int, lb::Nothing, ub::Nothing)
    MOI.Integer()
end

function _create_integer_domain(id::Int, lb::Vector, ub::Vector)
    (iszero(lb[id]) && isone(ub[id])) && return MOI.ZeroOne()
    return _create_integer_domain(id, nothing, nothing)
end

function SciMLBase.__solve(prob::OptimizationProblem,
                           opt::Union{MOI.AbstractOptimizer, MOI.OptimizerWithAttributes};
                           maxiters::Union{Number, Nothing} = nothing,
                           maxtime::Union{Number, Nothing} = nothing,
                           abstol::Union{Number, Nothing} = nothing,
                           reltol::Union{Number, Nothing} = nothing,
                           kwargs...)
    maxiters = Optimization._check_and_convert_maxiters(maxiters)
    maxtime = Optimization._check_and_convert_maxtime(maxtime)
    opt_setup = __map_optimizer_args(prob,
                                     opt;
                                     abstol = abstol,
                                     reltol = reltol,
                                     maxiters = maxiters,
                                     maxtime = maxtime,
                                     kwargs...)
    num_variables = length(prob.u0)
    θ = MOI.add_variables(opt_setup, num_variables)
    if prob.lb !== nothing
        @assert eachindex(prob.lb) == Base.OneTo(num_variables)
        for i in 1:num_variables
            if prob.lb[i] > -Inf
                MOI.add_constraint(opt_setup, θ[i], MOI.GreaterThan(prob.lb[i]))
            end
        end
    end
    if prob.ub !== nothing
        @assert eachindex(prob.ub) == Base.OneTo(num_variables)
        for i in 1:num_variables
            if prob.ub[i] < Inf
                MOI.add_constraint(opt_setup, θ[i], MOI.LessThan(prob.ub[i]))
            end
        end
    end

    if prob.int !== nothing
        @assert eachindex(prob.int) == Base.OneTo(num_variables)
        for i in 1:num_variables
            if prob.int[i]
                MOI.add_constraint(opt_setup, θ[i],
                                   _create_integer_domain(i, prob.lb, prob.ub))
            end
        end
    end

    if MOI.supports(opt_setup, MOI.VariablePrimalStart(), MOI.VariableIndex)
        @assert eachindex(prob.u0) == Base.OneTo(num_variables)
        for i in 1:num_variables
            MOI.set(opt_setup, MOI.VariablePrimalStart(), θ[i], prob.u0[i])
        end
    end
    MOI.set(opt_setup,
            MOI.ObjectiveSense(),
            prob.sense === Optimization.MaxSense ? MOI.MAX_SENSE : MOI.MIN_SENSE)
    if prob.lcons === nothing
        @assert prob.ucons === nothing
        con_bounds = MOI.NLPBoundsPair[]
    else
        @assert prob.ucons !== nothing
        con_bounds = MOI.NLPBoundsPair.(prob.lcons, prob.ucons)
    end
    MOI.set(opt_setup,
            MOI.NLPBlock(),
            MOI.NLPBlockData(con_bounds, MOIOptimizationProblem(prob), true))
    t0 = time()
    MOI.optimize!(opt_setup)
    t1 = time()
    if MOI.get(opt_setup, MOI.ResultCount()) >= 1
        minimizer = MOI.get(opt_setup, MOI.VariablePrimal(), θ)
        objective = MOI.get(opt_setup, MOI.ObjectiveValue())
        opt_ret = __moi_status_to_ReturnCode(MOI.get(opt_setup, MOI.TerminationStatus()))
    else
        minimizer = fill(NaN, num_variables)
        objective = NaN
        opt_ret = ReturnCode.Default
    end
    return SciMLBase.build_solution(SciMLBase.DefaultOptimizationCache(prob.f, prob.p),
                                    opt,
                                    minimizer,
                                    objective;
                                    original = opt_setup,
                                    retcode = opt_ret, solve_time = t1 - t0)
end

end
