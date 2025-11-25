module OptimizationMadNLP

using Reexport
@reexport using OptimizationBase
using OptimizationBase: MinSense, MaxSense, DEFAULT_CALLBACK
using MadNLP
using NLPModels
using SparseArrays

export MadNLPOptimizer

struct NLPModelsAdaptor{C, T, HB} <: NLPModels.AbstractNLPModel{T, Vector{T}}
    cache::C
    meta::NLPModels.NLPModelMeta{T, Vector{T}}
    counters::NLPModels.Counters
    jac_rows::Vector{Int}
    jac_cols::Vector{Int}
    jac_buffer::AbstractMatrix{T}
    hess_rows::Vector{Int}
    hess_cols::Vector{Int}
    hess_buffer::HB  # Can be Vector{T} or Matrix{T}
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
        # Create a values buffer matching the number of lower triangle elements
        hess_buffer = zeros(T, sum(lower_mask))
    elseif !isnothing(hess_proto)
        # Dense Hessian
        n = size(hess_proto, 1)
        hess_rows, hess_cols = _enumerate_lower_triangle(n)
        # For dense, store the full matrix but we'll extract values later
        hess_buffer = similar(hess_proto, T)
    else
        # No prototype - create dense structure
        n = meta.nvar
        hess_rows, hess_cols = _enumerate_lower_triangle(n)
        hess_buffer = zeros(T, n, n)
    end

    return NLPModelsAdaptor{C, T, typeof(hess_buffer)}(cache, meta, counters,
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
        if nlp.hess_buffer isa AbstractVector
            # For sparse prototypes, hess_buffer is already a values vector
            nlp.cache.f.lag_h(nlp.hess_buffer, x, obj_weight, y)
        else
            # For dense matrices, we need to pass the full matrix and extract values
            nlp.cache.f.lag_h(nlp.hess_buffer, x, obj_weight, y)
        end
    else
        # Manual computation: objective + constraint Hessians
        nlp.cache.f.hess(nlp.hess_buffer, x)
        nlp.hess_buffer .*= obj_weight

        if !isnothing(nlp.cache.f.cons_h) && !isempty(y)
            # Add weighted constraint Hessians
            cons_hessians = [similar(nlp.hess_buffer, eltype(nlp.hess_buffer))
                             for _ in 1:length(y)]
            nlp.cache.f.cons_h(cons_hessians, x)
            for (λ, H_cons) in zip(y, cons_hessians)
                nlp.hess_buffer .+= λ .* H_cons
            end
        end
    end

    if !isempty(H)
        # Extract values depending on buffer type
        if nlp.hess_buffer isa AbstractVector
            # For sparse, hess_buffer already contains just the values
            copyto!(H, nlp.hess_buffer)
        else
            # For dense matrices, extract lower triangle values
            for (idx, (i, j)) in enumerate(zip(nlp.hess_rows, nlp.hess_cols))
                H[idx] = nlp.hess_buffer[i, j]
            end
        end
    end

    return H
end

function NLPModels.jtprod!(
        nlp::NLPModelsAdaptor, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
    # Compute J^T * v using the AD-provided VJP (Vector-Jacobian Product)
    if !isnothing(nlp.cache.f.cons_vjp) && !isempty(Jtv)
        nlp.cache.f.cons_vjp(Jtv, x, v)
    end
    return Jtv
end

function NLPModels.jprod!(
        nlp::NLPModelsAdaptor, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
    # Compute J * v using the AD-provided JVP (Jacobian-Vector Product)
    if !isnothing(nlp.cache.f.cons_jvp) && !isempty(Jv)
        nlp.cache.f.cons_jvp(Jv, x, v)
    end
    return Jv
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

    # Initialization Options
    nlp_scaling::Bool = true
    nlp_scaling_max_gradient::Float64 = 100.0

    # Linear solver configuration
    linear_solver::Union{Nothing, Type} = nothing  # e.g., MumpsSolver, LapackCPUSolver, UmfpackSolver

    kkt_system::Union{Nothing, Type} = nothing # e.g. DenseKKTSystem

    mu_init::T = 0.1

    # Quasi-Newton options (used when hessian_approximation is CompactLBFGS, BFGS, or DampedBFGS)
    quasi_newton_options::Union{Nothing, MadNLP.QuasiNewtonOptions} = nothing

    # Additional MadNLP options
    additional_options::Dict{Symbol, Any} = Dict{Symbol, Any}()
end

SciMLBase.has_init(opt::MadNLPOptimizer) = true

SciMLBase.allowscallback(opt::MadNLPOptimizer) = false

function SciMLBase.requiresgradient(opt::MadNLPOptimizer)
    true
end
function SciMLBase.requireshessian(opt::MadNLPOptimizer)
    opt.hessian_approximation === MadNLP.ExactHessian
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
    opt.hessian_approximation === MadNLP.ExactHessian
end
function SciMLBase.requiresconshess(opt::MadNLPOptimizer)
    opt.hessian_approximation === MadNLP.ExactHessian
end
function SciMLBase.allowsconsvjp(opt::MadNLPOptimizer)
    true
end
function SciMLBase.allowsconsjvp(opt::MadNLPOptimizer)
    true
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

function _get_nnzj(f, ncon, nvar)
    jac_prototype = f.cons_jac_prototype

    if isnothing(jac_prototype)
        # No prototype - assume dense structure if there are constraints
        return ncon > 0 ? ncon * nvar : 0
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
        callback = DEFAULT_CALLBACK)
    nvar = length(cache.u0)
    ncon = !isnothing(cache.lcons) ? length(cache.lcons) : 0

    if !(callback isa OptimizationBase.NullCallback) || progress
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
        nnzj = _get_nnzj(cache.f, ncon, nvar),
        nnzh = _get_nnzh(cache.f, ncon, nvar),
        x0 = cache.u0,
        y0 = zeros(eltype(cache.u0), ncon),
        lvar,
        uvar,
        lcon,
        ucon,
        minimize = cache.sense !== MaxSense  # Default to minimization when sense is nothing or MinSense
    )

    if verbose isa Bool
        print_level = verbose ? MadNLP.INFO : MadNLP.WARN
    else
        print_level = verbose
    end

    !isnothing(reltol) && @warn "reltol not supported by MadNLP, use abstol instead."
    tol = isnothing(abstol) ? 1e-8 : abstol
    max_iter = isnothing(maxiters) ? 3000 : maxiters
    max_wall_time = isnothing(maxtime) ? 1e6 : maxtime

    # Build final options dictionary
    options = Dict{Symbol, Any}(opt.additional_options)

    options[:mu_init] = opt.mu_init

    # Add quasi_newton_options if provided, otherwise create default
    if !isnothing(opt.quasi_newton_options)
        options[:quasi_newton_options] = opt.quasi_newton_options
    else
        # Create default quasi-Newton options
        options[:quasi_newton_options] = MadNLP.QuasiNewtonOptions{T}()
    end

    # Add linear_solver if provided
    if !isnothing(opt.linear_solver)
        options[:linear_solver] = opt.linear_solver
    end

    if !isnothing(opt.kkt_system)
        options[:kkt_system] = opt.kkt_system
    end

    options[:rethrow_error] = opt.rethrow_error
    options[:disable_garbage_collector] = opt.disable_garbage_collector
    options[:blas_num_threads] = opt.blas_num_threads
    options[:output_file] = opt.output_file
    options[:file_print_level] = opt.file_print_level
    options[:acceptable_tol] = opt.acceptable_tol
    options[:acceptable_iter] = opt.acceptable_iter
    options[:jacobian_constant] = opt.jacobian_constant
    options[:hessian_constant] = opt.hessian_constant
    options[:hessian_approximation] = opt.hessian_approximation
    options[:nlp_scaling] = opt.nlp_scaling
    options[:nlp_scaling_max_gradient] = opt.nlp_scaling_max_gradient
    options[:print_level] = print_level
    options[:tol] = tol
    options[:max_iter] = max_iter
    options[:max_wall_time] = max_wall_time

    meta, options
end

function SciMLBase.__solve(cache::OptimizationCache{O}) where {O <: MadNLPOptimizer}
    maxiters = OptimizationBase._check_and_convert_maxiters(cache.solver_args.maxiters)
    maxtime = OptimizationBase._check_and_convert_maxtime(cache.solver_args.maxtime)
    maxtime = maxtime isa Float32 ? convert(Float64, maxtime) : maxtime

    meta, options = __map_optimizer_args(cache,
        cache.opt;
        abstol = cache.solver_args.abstol,
        reltol = cache.solver_args.reltol,
        maxiters = maxiters,
        maxtime = maxtime,
        verbose = get(cache.solver_args, :verbose, false),
        progress = cache.progress,
        callback = cache.callback
    )

    nlp = NLPModelsAdaptor(cache, meta, NLPModels.Counters())
    solver = MadNLP.MadNLPSolver(nlp; options...)
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
