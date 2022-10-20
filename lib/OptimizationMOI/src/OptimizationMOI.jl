module OptimizationMOI

using Reexport
@reexport using Optimization
using MathOptInterface
using Optimization.SciMLBase
using SparseArrays
using Setfield

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

struct MOIOptimizationCache{T, F <: OptimizationFunction, uType, P, LB, UB, I,
                              JT <: DenseOrSparse{T}, HT <: DenseOrSparse{T},
                              CHT <: DenseOrSparse{T}, S, PR} <:
       MOI.AbstractNLPEvaluator
    f::F
    u0::uType
    p::P
    lb::LB
    ub::UB
    int::I
    J::JT
    H::HT
    cons_H::Vector{CHT}
    lcons::Vector{T}
    ucons::Vector{T}
    sense::S
    prob::PR # TODO: this is dangerous since the cache and problem might get out of sync after reinit
end

function MOIOptimizationCache(prob::OptimizationProblem)
    num_cons = prob.ucons === nothing ? 0 : length(prob.ucons)
    f = Optimization.instantiate_function(prob.f, prob.u0, prob.f.adtype, prob.p, num_cons)
    T = eltype(prob.u0)
    n = length(prob.u0)
    return MOIOptimizationCache(f,
                                  prob.u0,
                                  prob.p,
                                  prob.lb,
                                  prob.ub, 
                                  prob.int,
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
                                  prob.ucons === nothing ? fill(Inf, num_cons) : prob.ucons,
                                  prob.sense,
                                  prob)
end

function MOI.features_available(cache::MOIOptimizationCache)
    features = [:Grad, :Hess, :Jac]
    # Assume that if there are constraints and expr then cons_expr exists
    if cache.f.expr !== nothing
        push!(features, :ExprGraph)
    end
    return features
end

function MOI.initialize(cache::MOIOptimizationCache,
                        requested_features::Vector{Symbol})
    available_features = MOI.features_available(cache)
    for feat in requested_features
        if !(feat in available_features)
            error("Unsupported feature $feat")
            # TODO: implement Jac-vec and Hess-vec products
            # for solvers that need them
        end
    end
    return
end

function MOI.eval_objective(cache::MOIOptimizationCache, x)
    return cache.f(x, cache.p)
end

function MOI.eval_constraint(cache::MOIOptimizationCache, g, x)
    cache.f.cons(g, x)
    return
end

function MOI.eval_objective_gradient(cache::MOIOptimizationCache, G, x)
    cache.f.grad(G, x)
    return
end

# This structure assumes the calculation of moiproblem.J is dense.
function MOI.jacobian_structure(cache::MOIOptimizationCache)
    if cache.J isa SparseMatrixCSC
        rows, cols, _ = findnz(cache.J)
        inds = Tuple{Int, Int}[(i, j) for (i, j) in zip(rows, cols)]
    else
        rows, cols = size(cache.J)
        inds = Tuple{Int, Int}[(i, j) for j in 1:cols for i in 1:rows]
    end
    return inds
end

function MOI.eval_constraint_jacobian(cache::MOIOptimizationCache, j, x)
    if isempty(j)
        return
    elseif cache.f.cons_j === nothing
        error("Use OptimizationFunction to pass the derivatives or " *
              "automatically generate them with one of the autodiff backends")
    end
    cache.f.cons_j(cache.J, x)
    if cache.J isa SparseMatrixCSC
        nnz = nonzeros(cache.J)
        @assert length(j) == length(nnz)
        for (i, Ji) in zip(eachindex(j), nnz)
            j[i] = Ji
        end
    else
        for i in eachindex(j)
            j[i] = cache.J[i]
        end
    end
    return
end

function MOI.hessian_lagrangian_structure(cache::MOIOptimizationCache)
    sparse_obj = cache.H isa SparseMatrixCSC
    sparse_constraints = all(H -> H isa SparseMatrixCSC, cache.cons_H)
    if !sparse_constraints && any(H -> H isa SparseMatrixCSC, cache.cons_H)
        # Some constraint hessians are dense and some are sparse! :(
        error("Mix of sparse and dense constraint hessians are not supported")
    end
    N = length(cache.u0)
    inds = if sparse_obj
        rows, cols, _ = findnz(cache.H)
        Tuple{Int, Int}[(i, j) for (i, j) in zip(rows, cols) if i <= j]
    else
        Tuple{Int, Int}[(row, col) for col in 1:N for row in 1:col]
    end
    if sparse_constraints
        for Hi in cache.cons_H
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

function MOI.eval_hessian_lagrangian(cache::MOIOptimizationCache{T},
                                     h,
                                     x,
                                     σ,
                                     μ) where {T}
    fill!(h, zero(T))
    k = 0
    cache.f.hess(cache.H, x)
    sparse_objective = cache.H isa SparseMatrixCSC
    if sparse_objective
        rows, cols, _ = findnz(cache.H)
        for (i, j) in zip(rows, cols)
            if i <= j
                k += 1
                h[k] = σ * cache.H[i, j]
            end
        end
    else
        for i in 1:size(cache.H, 1), j in 1:i
            k += 1
            h[k] = σ * cache.H[i, j]
        end
    end
    # A count of the number of non-zeros in the objective Hessian is needed if
    # the constraints are dense.
    nnz_objective = k
    if !isempty(μ) && !all(iszero, μ)
        cache.f.cons_h(cache.cons_H, x)
        for (μi, Hi) in zip(μ, cache.cons_H)
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

function MOI.objective_expr(cache::MOIOptimizationCache)
    return _replace_variable_indices(cache.f.expr)
end

function MOI.constraint_expr(cache::MOIOptimizationCache, i)
    # expr has the form f(x) == 0
    expr = _replace_variable_indices(cache.f.cons_expr[i].args[2])
    lb, ub = cache.lcons[i], cache.ucons[i]
    return :($lb <= $expr <= $ub)
end

function _create_new_optimizer(cache::MOI.OptimizerWithAttributes)
    return _create_new_optimizer(MOI.instantiate(cache))
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

function __map_optimizer_args(cache::MOIOptimizationCache,
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

# TODO: this needs to go but needs a change in SciMLBase/solve.jl
function SciMLBase.solve(prob::OptimizationProblem,
                           opt::Union{MOI.AbstractOptimizer, MOI.OptimizerWithAttributes},
                           args...;
                           kwargs...)
    cache = SciMLBase.init(prob, opt, args...; kwargs...)
    SciMLBase.solve!(cache, opt, args...; kwargs...)
end

function SciMLBase.init(prob::OptimizationProblem, args...; kwargs...) 
    cache = MOIOptimizationCache(prob)
    return cache
end

function SciMLBase.solve!(cache::MOIOptimizationCache,
    opt::Union{MOI.AbstractOptimizer, MOI.OptimizerWithAttributes};
    maxiters::Union{Number, Nothing} = nothing,
    maxtime::Union{Number, Nothing} = nothing,
    abstol::Union{Number, Nothing} = nothing,
    reltol::Union{Number, Nothing} = nothing,
    kwargs...)

    maxiters = Optimization._check_and_convert_maxiters(maxiters)
    maxtime = Optimization._check_and_convert_maxtime(maxtime)
    opt_setup = __map_optimizer_args(cache,
                                     opt;
                                     abstol = abstol,
                                     reltol = reltol,
                                     maxiters = maxiters,
                                     maxtime = maxtime,
                                     kwargs...)
    num_variables = length(cache.u0)
    θ = MOI.add_variables(opt_setup, num_variables)
    if cache.lb !== nothing
        @assert eachindex(cache.lb) == Base.OneTo(num_variables)
    end
    if cache.ub !== nothing
        @assert eachindex(cache.ub) == Base.OneTo(num_variables)
    end

    for i in 1:num_variables
        if cache.lb !== nothing && cache.lb[i] > -Inf
            MOI.add_constraint(opt_setup, θ[i], MOI.GreaterThan(cache.lb[i]))
        end
        if cache.ub !== nothing && cache.ub[i] < Inf
            MOI.add_constraint(opt_setup, θ[i], MOI.LessThan(cache.ub[i]))
        end
        if cache.int !== nothing && cache.int[i]
            if cache.lb !== nothing && cache.lb[i] == 0 && cache.ub !== nothing && cache.ub[i] == 1
                MOI.add_constraint(opt_setup, θ[i], MOI.ZeroOne())
            else
                MOI.add_constraint(opt_setup, θ[i], MOI.Integer())
            end
        end
    end

    if MOI.supports(opt_setup, MOI.VariablePrimalStart(), MOI.VariableIndex)
        @assert eachindex(cache.u0) == Base.OneTo(num_variables)
        for i in 1:num_variables
            MOI.set(opt_setup, MOI.VariablePrimalStart(), θ[i], cache.u0[i])
        end
    end

    MOI.set(opt_setup,
            MOI.ObjectiveSense(),
            cache.sense === Optimization.MaxSense ? MOI.MAX_SENSE : MOI.MIN_SENSE)
    if cache.lcons === nothing
        @assert cache.ucons === nothing
        con_bounds = MOI.NLPBoundsPair[]
    else
        @assert cache.ucons !== nothing
        con_bounds = MOI.NLPBoundsPair.(cache.lcons, cache.ucons)
    end
    MOI.set(opt_setup,
            MOI.NLPBlock(),
            MOI.NLPBlockData(con_bounds, cache, true))
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
    return SciMLBase.build_solution(cache.prob,
                                    opt,
                                    minimizer,
                                    minimum;
                                    original = opt_setup,
                                    retcode = opt_ret)
end

function SciMLBase.reinit!(cache::MOIOptimizationCache; p=nothing)
    if !isnothing(p)
        @set! cache.p = p
    end
    return cache
end

end
