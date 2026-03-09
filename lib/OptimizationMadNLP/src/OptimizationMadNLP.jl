module OptimizationMadNLP

using Reexport
@reexport using OptimizationBase
using OptimizationBase: MinSense, MaxSense, DEFAULT_CALLBACK
using OptimizationNLPModels: NLPModelsAdaptor, build_nlpmodel_meta
using MadNLP
using NLPModels
using SparseArrays

export MadNLPOptimizer

include("callback.jl")

@kwdef struct MadNLPOptimizer{T}
    # General options
    rethrow_error::Bool = true
    disable_garbage_collector::Bool = false
    blas_num_threads::Int = 1

    # Output options
    output_file::String = ""
    file_print_level::MadNLP.LogLevels = MadNLP.INFO

    # Termination options
    acceptable_tol::T = 1.0e-6
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

    # Barrier update strategy (e.g., MonotoneUpdate, QualityFunctionUpdate, LOQOUpdate)
    # Each barrier struct has its own mu_init field.
    barrier::Union{Nothing, MadNLP.AbstractBarrierUpdate} = nothing

    # Quasi-Newton options (used when hessian_approximation is CompactLBFGS, BFGS, or DampedBFGS)
    quasi_newton_options::Union{Nothing, MadNLP.QuasiNewtonOptions} = nothing

    # Additional MadNLP options
    additional_options::Dict{Symbol, Any} = Dict{Symbol, Any}()
end

SciMLBase.has_init(opt::MadNLPOptimizer) = true

SciMLBase.allowscallback(opt::MadNLPOptimizer) = true

function SciMLBase.requiresgradient(opt::MadNLPOptimizer)
    return true
end
function SciMLBase.requireshessian(opt::MadNLPOptimizer)
    return opt.hessian_approximation === MadNLP.ExactHessian
end
function SciMLBase.allowsbounds(opt::MadNLPOptimizer)
    return true
end
function SciMLBase.allowsconstraints(opt::MadNLPOptimizer)
    return true
end
function SciMLBase.requiresconsjac(opt::MadNLPOptimizer)
    return true
end
function SciMLBase.requireslagh(opt::MadNLPOptimizer)
    return opt.hessian_approximation === MadNLP.ExactHessian
end
function SciMLBase.requiresconshess(opt::MadNLPOptimizer)
    return opt.hessian_approximation === MadNLP.ExactHessian
end
function SciMLBase.allowsconsvjp(opt::MadNLPOptimizer)
    return true
end
function SciMLBase.allowsconsjvp(opt::MadNLPOptimizer)
    return true
end

function map_madnlp_status(status::MadNLP.Status)
    if status in [
            MadNLP.SOLVE_SUCCEEDED,
            MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL,
            MadNLP.USER_REQUESTED_STOP,
        ]
        return SciMLBase.ReturnCode.Success
    elseif status in [
            MadNLP.INFEASIBLE_PROBLEM_DETECTED,
            MadNLP.SEARCH_DIRECTION_BECOMES_TOO_SMALL,
            MadNLP.DIVERGING_ITERATES,
            MadNLP.RESTORATION_FAILED,
            MadNLP.NOT_ENOUGH_DEGREES_OF_FREEDOM,
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

function __map_optimizer_args(
        cache,
        opt::MadNLPOptimizer;
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        verbose = false,
        progress::Bool = false,
        callback = DEFAULT_CALLBACK
    )
    cb = MadNLPProgressLogger(callback, progress, maxiters, cache.p)

    T = eltype(cache.u0)

    meta = build_nlpmodel_meta(cache)

    if verbose isa Bool
        print_level = verbose ? MadNLP.INFO : MadNLP.WARN
    else
        print_level = verbose
    end

    !isnothing(reltol) && @SciMLMessage(
        "reltol not supported by MadNLP, use abstol instead.",
        cache.verbose, :unsupported_kwargs
    )
    tol = isnothing(abstol) ? 1.0e-8 : abstol
    max_iter = isnothing(maxiters) ? 3000 : maxiters
    max_wall_time = isnothing(maxtime) ? 1.0e6 : maxtime

    # Build final options dictionary
    options = Dict{Symbol, Any}(opt.additional_options)

    if !isnothing(opt.barrier)
        options[:barrier] = opt.barrier
    end

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
    options[:intermediate_callback] = cb

    return meta, options
end

function SciMLBase.__solve(cache::OptimizationCache{O}) where {O <: MadNLPOptimizer}
    maxiters = OptimizationBase._check_and_convert_maxiters(cache.solver_args.maxiters)
    maxtime = OptimizationBase._check_and_convert_maxtime(cache.solver_args.maxtime)
    maxtime = maxtime isa Float32 ? convert(Float64, maxtime) : maxtime

    meta, options = __map_optimizer_args(
        cache,
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

    if cache.progress
        Base.@logmsg Base.LogLevel(-1) progress="done" _id = :OptimizationMadNLP
    end

    stats = OptimizationBase.OptimizationStats(;
        time = results.counters.total_time,
        iterations = results.iter,
        fevals = results.counters.obj_cnt,
        gevals = results.counters.obj_grad_cnt
    )

    retcode = map_madnlp_status(results.status)

    return SciMLBase.build_solution(
        cache,
        cache.opt,
        results.solution,
        results.objective;
        original = results,
        retcode,
        stats
    )
end

end
