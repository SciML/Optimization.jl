module OptimizationMadNLP

using OptimizationBase
using MadNLP
using NLPModels
using SparseArrays

export MadNLPOptimizer

struct NLPModelsAdaptor{C, T} <: NLPModels.AbstractNLPModel{T, Vector{T}}
    cache::C
    meta::NLPModels.NLPModelMeta{T, Vector{T}}
    counters::NLPModels.Counters
    jac_rows::Vector{Int}
    jac_cols::Vector{Int}
    jac_buffer::AbstractMatrix{T}
    hess_rows::Vector{Int}
    hess_cols::Vector{Int}
    hess_buffer::AbstractMatrix{T}
end

function _enumerate_dense_structure(ncon, nvar)
    nnz = ncon * nvar
    rows = Vector{Int}(undef, nnz)
    cols = Vector{Int}(undef, nnz)
    idx = 1
    for j in 1:nvar
        for i in 1:ncon
            rows[idx] = i
            cols[idx] = j
            idx += 1
        end
    end
    return rows, cols
end

function _enumerate_lower_triangle(n)
    nnz = div(n * (n + 1), 2)
    rows = Vector{Int}(undef, nnz)
    cols = Vector{Int}(undef, nnz)
    idx = 1
    for j in 1:n
        for i in j:n  # Only lower triangle
            rows[idx] = i
            cols[idx] = j
            idx += 1
        end
    end
    return rows, cols
end

function NLPModelsAdaptor(
        cache::C, meta::NLPModels.NLPModelMeta{T, Vector{T}}, counters) where {C, T}
    # Extract Jacobian structure once
    jac_prototype = cache.f.cons_jac_prototype

    if jac_prototype isa SparseMatrixCSC
        jac_rows, jac_cols, _ = findnz(jac_prototype)
        jac_buffer = similar(jac_prototype)
    elseif jac_prototype isa AbstractMatrix
        ncon, nvar = size(jac_prototype)
        jac_rows, jac_cols = _enumerate_dense_structure(ncon, nvar)
        jac_buffer = similar(jac_prototype)
    else
        # Fallback: assume dense structure
        ncon, nvar = meta.ncon, meta.nvar
        jac_rows, jac_cols = _enumerate_dense_structure(ncon, nvar)
        jac_buffer = zeros(T, ncon, nvar)
    end

    ncon = !isnothing(cache.lcons) ? length(cache.lcons) : 0

    # Extract Hessian structure
    hess_proto = ncon > 0 ? cache.f.lag_hess_prototype : cache.f.hess_prototype

    if !isnothing(hess_proto) && hess_proto isa SparseMatrixCSC
        I, J, _ = findnz(hess_proto)
        # Keep only lower triangle
        lower_mask = I .>= J
        hess_rows = I[lower_mask]
        hess_cols = J[lower_mask]
        hess_buffer = similar(hess_proto)
    elseif !isnothing(hess_proto)
        # Dense Hessian
        n = size(hess_proto, 1)
        hess_rows, hess_cols = _enumerate_lower_triangle(n)
        hess_buffer = similar(hess_proto)
    else
        # No prototype - create dense structure
        n = meta.nvar
        hess_rows, hess_cols = _enumerate_lower_triangle(n)
        hess_buffer = zeros(T, n, n)
    end

    return NLPModelsAdaptor{C, T}(cache, meta, counters,
        jac_rows, jac_cols, jac_buffer,
        hess_rows, hess_cols, hess_buffer)
end

function NLPModels.obj(nlp::NLPModelsAdaptor, x::AbstractVector)
    nlp.cache.f(x, nlp.cache.p)
end

function NLPModels.grad!(nlp::NLPModelsAdaptor, x::AbstractVector, g::AbstractVector)
    nlp.cache.f.grad(g, x, nlp.cache.p)
    return g
end

function NLPModels.cons!(nlp::NLPModelsAdaptor, x::AbstractVector, c::AbstractVector)
    if !isempty(c)
        nlp.cache.f.cons(c, x)
    end
    return c
end

function NLPModels.jac_structure!(
        nlp::NLPModelsAdaptor, I::AbstractVector{T}, J::AbstractVector{T}) where {T}
    copyto!(I, nlp.jac_rows)
    copyto!(J, nlp.jac_cols)
    return I, J
end

function NLPModels.jac_coord!(
        nlp::NLPModelsAdaptor, x::AbstractVector, vals::AbstractVector)
    if !isempty(vals)
        # Evaluate Jacobian into preallocated buffer
        nlp.cache.f.cons_j(nlp.jac_buffer, x)

        # Extract values in COO order
        if nlp.jac_buffer isa SparseMatrixCSC
            _, _, v = findnz(nlp.jac_buffer)
            copyto!(vals, v)
        else
            # Dense case: extract in column-major order matching structure
            for (idx, (i, j)) in enumerate(zip(nlp.jac_rows, nlp.jac_cols))
                vals[idx] = nlp.jac_buffer[i, j]
            end
        end
    end

    return vals
end

function NLPModels.hess_structure!(
        nlp::NLPModelsAdaptor, I::AbstractVector{T}, J::AbstractVector{T}) where {T}
    copyto!(I, nlp.hess_rows)
    copyto!(J, nlp.hess_cols)
    return I, J
end

function NLPModels.hess_coord!(
        nlp::NLPModelsAdaptor, x, y, H::AbstractVector; obj_weight = 1.0)
    if !isnothing(nlp.cache.f.lag_h)
        # Use Lagrangian Hessian directly
        nlp.cache.f.lag_h(nlp.hess_buffer, x, obj_weight, y)
    else
        # Manual computation: objective + constraint Hessians
        nlp.cache.f.hess(nlp.hess_buffer, x)
        nlp.hess_buffer .*= obj_weight

        if !isnothing(nlp.cache.f.cons_h) && !isempty(y)
            # Add weighted constraint Hessians
            cons_hessians = [similar(nlp.hess_buffer) for _ in 1:length(y)]
            nlp.cache.f.cons_h(cons_hessians, x)
            for (λ, H_cons) in zip(y, cons_hessians)
                nlp.hess_buffer .+= λ .* H_cons
            end
        end
    end

    if !isempty(H)
        # Extract lower triangle values
        for (idx, (i, j)) in enumerate(zip(nlp.hess_rows, nlp.hess_cols))
            H[idx] = nlp.hess_buffer[i, j]
        end
    end

    return H
end

@kwdef struct MadNLPOptimizer{T}
    # General options
    rethrow_error::Bool = true
    disable_garbage_collector::Bool = false
    blas_num_threads::Int = 1

    # Output options
    output_file::String = ""
    file_print_level::MadNLP.LogLevels = MadNLP.INFO

    # Termination options
    acceptable_tol::T = 1e-6
    acceptable_iter::Int = 15

    # NLP options
    jacobian_constant::Bool = false
    hessian_constant::Bool = false
    hessian_approximation::Type = MadNLP.ExactHessian

    # Barrier
    mu_init::T = 1e-1

    # Additional MadNLP options
    additional_options::Dict{Symbol, Any} = Dict{Symbol, Any}()
end

SciMLBase.supports_opt_cache_interface(opt::MadNLPOptimizer) = true

function SciMLBase.requiresgradient(opt::MadNLPOptimizer)
    true
end
function SciMLBase.requireshessian(opt::MadNLPOptimizer)
    true
end
function SciMLBase.allowsbounds(opt::MadNLPOptimizer)
    true
end
function SciMLBase.allowsconstraints(opt::MadNLPOptimizer)
    true
end
function SciMLBase.requiresconsjac(opt::MadNLPOptimizer)
    true
end
function SciMLBase.requireslagh(opt::MadNLPOptimizer)
    true
end

function SciMLBase.__init(prob::SciMLBase.OptimizationProblem, opt::MadNLPOptimizer;
        kwargs...)
    return OptimizationCache(prob, opt; kwargs...)
end

function map_madnlp_status(status::MadNLP.Status)
    if status in [
        MadNLP.SOLVE_SUCCEEDED,
        MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL,
        MadNLP.USER_REQUESTED_STOP
    ]
        return SciMLBase.ReturnCode.Success
    elseif status in [
        MadNLP.INFEASIBLE_PROBLEM_DETECTED,
        MadNLP.SEARCH_DIRECTION_BECOMES_TOO_SMALL,
        MadNLP.DIVERGING_ITERATES,
        MadNLP.RESTORATION_FAILED,
        MadNLP.NOT_ENOUGH_DEGREES_OF_FREEDOM
    ]
        return SciMLBase.ReturnCode.Infeasible
    elseif status == MadNLP.MAXIMUM_ITERATIONS_EXCEEDED
        return SciMLBase.ReturnCode.MaxIters
    elseif status == MadNLP.MAXIMUM_WALLTIME_EXCEEDED
        return SciMLBase.ReturnCode.MaxTime
    else
        # All error codes and invalid numbers
        return SciMLBase.ReturnCode.Failure
    end
end

function _get_nnzj(f)
    jac_prototype = f.cons_jac_prototype

    if isnothing(jac_prototype)
        return 0
    elseif jac_prototype isa SparseMatrixCSC
        nnz(jac_prototype)
    else
        prod(size(jac_prototype))
    end
end

function _get_nnzh(f, ncon, nvar)
    # For constrained problems, use Lagrangian Hessian; for unconstrained, use objective Hessian
    hess_proto = ncon > 0 ? f.lag_hess_prototype : f.hess_prototype

    if isnothing(hess_proto)
        # No prototype provided - assume dense Hessian
        return div(nvar * (nvar + 1), 2)
    elseif hess_proto isa SparseMatrixCSC
        # Only count lower triangle
        I, J, _ = findnz(hess_proto)
        return count(i >= j for (i, j) in zip(I, J))
    else
        # Dense: n(n+1)/2
        n = size(hess_proto, 1)
        return div(n * (n + 1), 2)
    end
end

function __map_optimizer_args(cache,
        opt::MadNLPOptimizer;
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        verbose = false,
        progress::Bool = false,
        callback = nothing)
    nvar = length(cache.u0)
    ncon = !isnothing(cache.lcons) ? length(cache.lcons) : 0

    if !isnothing(progress) || !isnothing(callback)
        @warn("MadNLP doesn't currently support user defined callbacks.")
    end
    # TODO: add support for user callbacks in MadNLP

    T = eltype(cache.u0)
    lvar = something(cache.lb, fill(-Inf, nvar))
    uvar = something(cache.ub, fill(Inf, nvar))
    lcon = something(cache.lcons, T[])
    ucon = something(cache.ucons, T[])

    meta = NLPModels.NLPModelMeta(
        nvar;
        ncon,
        nnzj = _get_nnzj(cache.f),
        nnzh = _get_nnzh(cache.f, ncon, nvar),
        x0 = cache.u0,
        y0 = zeros(eltype(cache.u0), ncon),
        lvar,
        uvar,
        lcon,
        ucon,
        minimize = cache.sense == MinSense
    )

    nlp = NLPModelsAdaptor(cache, meta, NLPModels.Counters())

    if verbose isa Bool
        print_level = verbose ? MadNLP.INFO : MadNLP.WARN
    else
        print_level = verbose
    end
    # use MadNLP defaults
    tol = isnothing(reltol) ? 1e-8 : reltol
    max_iter = isnothing(maxiters) ? 3000 : maxiters
    max_wall_time = isnothing(maxtime) ? 1e6 : maxtime

    MadNLP.MadNLPSolver(nlp;
        print_level, tol, max_iter, max_wall_time,
        opt.rethrow_error,
        opt.disable_garbage_collector,
        opt.blas_num_threads,
        opt.output_file,
        opt.file_print_level,
        opt.acceptable_tol,
        opt.acceptable_iter,
        opt.jacobian_constant,
        opt.hessian_constant,
        opt.hessian_approximation,
        opt.mu_init,
        opt.additional_options...
    )
end

function SciMLBase.__solve(cache::OptimizationBase.OptimizationCache{
        F,
        RC,
        LB,
        UB,
        LC,
        UC,
        S,
        O,
        D,
        P,
        C
}) where {
        F,
        RC,
        LB,
        UB,
        LC,
        UC,
        S,
        O <: MadNLPOptimizer,
        D,
        P,
        C
}
    maxiters = OptimizationBase._check_and_convert_maxiters(cache.solver_args.maxiters)
    maxtime = OptimizationBase._check_and_convert_maxtime(cache.solver_args.maxtime)

    solver = __map_optimizer_args(cache,
        cache.opt;
        abstol = cache.solver_args.abstol,
        reltol = cache.solver_args.reltol,
        maxiters = maxiters,
        maxtime = maxtime,
        verbose = get(cache.solver_args, :verbose, false),
        progress = cache.progress,
        callback = cache.callback
    )

    results = MadNLP.solve!(solver)

    stats = OptimizationBase.OptimizationStats(; time = results.counters.total_time,
        iterations = results.iter,
        fevals = results.counters.obj_cnt,
        gevals = results.counters.obj_grad_cnt)

    retcode = map_madnlp_status(results.status)

    return SciMLBase.build_solution(cache,
        cache.opt,
        results.solution,
        results.objective;
        original = results,
        retcode,
        stats)
end

end
