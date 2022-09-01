module OptimizationMOI

using MathOptInterface, Optimization, Optimization.SciMLBase, SparseArrays
const MOI = MathOptInterface

const DenseOrSparse{T} = Union{Matrix{T}, SparseMatrixCSC{T}}

struct MOIOptimizationProblem{T, F <: OptimizationFunction, uType, P,
                              JT <: DenseOrSparse{T},
                              LHT <: DenseOrSparse{T}} <:
       MOI.AbstractNLPEvaluator
    f::F
    u0::uType
    p::P
    J::JT
    lag_H::LHT
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
                                  isnothing(f.lag_hess_prototype) ? zeros(T, n, n) :
                                  convert.(T, f.lag_hess_prototype),
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
    sparse_lag = moiproblem.lag_H isa SparseMatrixCSC
    N = length(moiproblem.u0)
    inds = if sparse_lag
        rows, cols, _ = findnz(moiproblem.lag_H)
        Tuple{Int, Int}[(i, j) for (i, j) in zip(rows, cols) if i >= j]
    else
        Tuple{Int, Int}[(row, col) for col in 1:N for row in 1:col]
    end
    inds
end

function MOI.eval_hessian_lagrangian(moiproblem::MOIOptimizationProblem{T},
                                     h,
                                     x,
                                     σ,
                                     μ) where {T}
    fill!(h, zero(T))
    k = 0
    hess = moiproblem.H
    moiproblem.f.lag_h(hess, x, σ, μ)
    if hess isa SparseMatrixCSC
        for col in 1:size(hess, 2)
            for r in nzrange(hess, col)
                I = rowvals(hess)[r]
                J = col
                V = nonzeros(hess)[r]
                I >= J || continue
                h[k] = V
                k += 1
            end
        end
    else
        for i in 1:size(hess, 1), j in 1:i
            k += 1
            h[k] = hess[i, j]
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
        @warn "common reltol argument is currently not used by $(optimizer). Set tolerances via optimizer specific keyword aguments."
    end
    if !isnothing(abstol)
        @warn "common abstol argument is currently not used by $(optimizer). Set tolerances via optimizer specific keyword aguments."
    end
    if !isnothing(maxiters)
        @warn "common maxiters argument is currently not used by $(optimizer). Set number of interations via optimizer specific keyword aguments."
    end
    return optimizer
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
    MOI.optimize!(opt_setup)
    if MOI.get(opt_setup, MOI.ResultCount()) >= 1
        minimizer = MOI.get(opt_setup, MOI.VariablePrimal(), θ)
        minimum = MOI.get(opt_setup, MOI.ObjectiveValue())
        opt_ret = Symbol(string(MOI.get(opt_setup, MOI.TerminationStatus())))
    else
        minimizer = fill(NaN, num_variables)
        minimum = NaN
        opt_ret = :Default
    end
    return SciMLBase.build_solution(prob,
                                    opt,
                                    minimizer,
                                    minimum;
                                    original = opt_setup,
                                    retcode = opt_ret)
end

end
