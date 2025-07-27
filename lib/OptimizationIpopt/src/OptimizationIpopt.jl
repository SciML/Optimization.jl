module OptimizationIpopt

using Optimization
using Ipopt
using LinearAlgebra
using SparseArrays
using SciMLBase
using SymbolicIndexingInterface

export IpoptOptimizer

const DenseOrSparse{T} = Union{Matrix{T}, SparseMatrixCSC{T}}

struct IpoptOptimizer end

function SciMLBase.supports_opt_cache_interface(alg::IpoptOptimizer)
    true
end

function SciMLBase.requiresgradient(opt::IpoptOptimizer)
    true
end
function SciMLBase.requireshessian(opt::IpoptOptimizer)
    true
end
function SciMLBase.requiresconsjac(opt::IpoptOptimizer)
    true
end
function SciMLBase.requiresconshess(opt::IpoptOptimizer)
    true
end

function SciMLBase.allowsbounds(opt::IpoptOptimizer)
    true
end
function SciMLBase.allowsconstraints(opt::IpoptOptimizer)
    true
end

include("cache.jl")
include("callback.jl")

function __map_optimizer_args(cache,
        opt::IpoptOptimizer;
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        hessian_approximation = "exact",
        verbose = false,
        progress = false,
        callback = nothing,
        kwargs...)
    jacobian_sparsity = jacobian_structure(cache)
    hessian_sparsity = hessian_lagrangian_structure(cache)

    eval_f(x) = eval_objective(cache, x)
    eval_grad_f(x, grad_f) = eval_objective_gradient(cache, grad_f, x)
    eval_g(x, g) = eval_constraint(cache, g, x)
    function eval_jac_g(x, rows, cols, values)
        if values === nothing
            for i in 1:length(jacobian_sparsity)
                rows[i], cols[i] = jacobian_sparsity[i]
            end
        else
            eval_constraint_jacobian(cache, values, x)
        end
        return
    end
    function eval_h(x, rows, cols, obj_factor, lambda, values)
        if values === nothing
            for i in 1:length(hessian_sparsity)
                rows[i], cols[i] = hessian_sparsity[i]
            end
        else
            eval_hessian_lagrangian(cache, values, x, obj_factor, lambda)
        end
        return
    end

    lb = isnothing(cache.lb) ? fill(-Inf, cache.n) : cache.lb
    ub = isnothing(cache.ub) ? fill(Inf, cache.n) : cache.ub

    prob = Ipopt.CreateIpoptProblem(
        cache.n,
        lb,
        ub,
        cache.num_cons,
        cache.lcons,
        cache.ucons,
        length(jacobian_structure(cache)),
        length(hessian_lagrangian_structure(cache)),
        eval_f,
        eval_g,
        eval_grad_f,
        eval_jac_g,
        eval_h
    )
    progress_callback = IpoptProgressLogger(cache.progress, cache, prob)
    intermediate = (args...) -> progress_callback(args...)
    Ipopt.SetIntermediateCallback(prob, intermediate)

    if !isnothing(maxiters)
        Ipopt.AddIpoptIntOption(prob, "max_iter", maxiters)
    end
    if !isnothing(maxtime)
        Ipopt.AddIpoptNumOption(prob, "max_cpu_time", maxtime)
    end
    if !isnothing(reltol)
        Ipopt.AddIpoptNumOption(prob, "tol", reltol)
    end
    if verbose isa Bool
        Ipopt.AddIpoptIntOption(prob, "print_level", verbose * 5)
    else
        Ipopt.AddIpoptIntOption(prob, "print_level", verbose)
    end
    Ipopt.AddIpoptStrOption(prob, "hessian_approximation", hessian_approximation)

    for kw in pairs(kwargs)
        if kw[2] isa Int
            Ipopt.AddIpoptIntOption(prob, string(kw[1]), kw[2])
        elseif kw[2] isa Float64
            Ipopt.AddIpoptNumOption(prob, string(kw[1]), kw[2])
        elseif kw[2] isa String
            Ipopt.AddIpoptStrOption(prob, string(kw[1]), kw[2])
        else
            error("Keyword argument type $(typeof(kw[2])) not recognized")
        end
    end

    return prob
end

function map_retcode(solvestat)
    status = Ipopt.ApplicationReturnStatus(solvestat)
    if status in [
        Ipopt.Solve_Succeeded,
        Ipopt.Solved_To_Acceptable_Level,
        Ipopt.User_Requested_Stop,
        Ipopt.Feasible_Point_Found
    ]
        return ReturnCode.Success
    elseif status in [
        Ipopt.Infeasible_Problem_Detected,
        Ipopt.Search_Direction_Becomes_Too_Small,
        Ipopt.Diverging_Iterates
    ]
        return ReturnCode.Infeasible
    elseif status == Ipopt.Maximum_Iterations_Exceeded
        return ReturnCode.MaxIters
    elseif status in [Ipopt.Maximum_CpuTime_Exceeded
                      Ipopt.Maximum_WallTime_Exceeded]
        return ReturnCode.MaxTime
    else
        return ReturnCode.Failure
    end
end

function SciMLBase.__solve(cache::IpoptCache)
    maxiters = Optimization._check_and_convert_maxiters(cache.solver_args.maxiters)
    maxtime = Optimization._check_and_convert_maxtime(cache.solver_args.maxtime)

    opt_setup = __map_optimizer_args(cache,
        cache.opt;
        abstol = cache.solver_args.abstol,
        reltol = cache.solver_args.reltol,
        maxiters = maxiters,
        maxtime = maxtime,
        cache.solver_args...)

    opt_setup.x .= cache.reinit_cache.u0

    start_time = time()
    status = Ipopt.IpoptSolve(opt_setup)

    opt_ret = map_retcode(status)

    if cache.progress
        # Set progressbar to 1 to finish it
        Base.@logmsg(Base.LogLevel(-1), "", progress=1, _id=:OptimizationIpopt)
    end

    minimum = opt_setup.obj_val
    minimizer = opt_setup.x

    stats = Optimization.OptimizationStats(; time = time() - start_time,
        iterations = cache.iterations, fevals = cache.f_calls, gevals = cache.f_grad_calls)

    finalize(opt_setup)

    return SciMLBase.build_solution(cache,
        cache.opt,
        minimizer,
        minimum;
        original = opt_setup,
        retcode = opt_ret,
        stats = stats)
end

function SciMLBase.__init(prob::OptimizationProblem,
        opt::IpoptOptimizer;
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        kwargs...)
    cache = IpoptCache(prob, opt;
        maxiters,
        maxtime,
        abstol,
        reltol,
        kwargs...
    )
    cache.reinit_cache.u0 .= prob.u0

    return cache
end

end # OptimizationIpopt
