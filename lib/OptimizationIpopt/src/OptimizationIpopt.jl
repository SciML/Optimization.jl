module OptimizationIpopt

using Reexport
@reexport using OptimizationBase
using Ipopt
using LinearAlgebra
using SparseArrays
using SciMLBase
using SymbolicIndexingInterface

export IpoptOptimizer

"""
    IpoptOptimizer(; kwargs...)

Optimizer using the Interior Point Optimizer (Ipopt) for nonlinear OptimizationBase.

Ipopt is designed to find (local) solutions of mathematical optimization problems of the form:

    min f(x)
    s.t. g_L ≤ g(x) ≤ g_U
         x_L ≤ x ≤ x_U

where f(x) and g(x) are twice continuously differentiable functions.

# Common Interface Arguments

The following common optimization arguments can be passed to `solve`:
- `reltol`: Overrides the Ipopt `tol` option (desired convergence tolerance)
- `maxiters`: Overrides the Ipopt `max_iter` option (maximum iterations)
- `maxtime`: Overrides the Ipopt `max_wall_time` option (maximum wall clock time)
- `verbose`: Overrides the Ipopt `print_level` option (0 for silent, 5 for default, up to 12 for maximum verbosity)

# Keyword Arguments

## Termination Options
- `acceptable_tol::Float64 = 1e-6`: Acceptable convergence tolerance (relative)
- `acceptable_iter::Int = 15`: Number of acceptable iterations before termination
- `dual_inf_tol::Float64 = 1.0`: Desired threshold for dual infeasibility
- `constr_viol_tol::Float64 = 1e-4`: Desired threshold for constraint violation
- `compl_inf_tol::Float64 = 1e-4`: Desired threshold for complementarity conditions

## Output Options
- `print_timing_statistics::String = "no"`: Print timing statistics at end of optimization
- `print_info_string::String = "no"`: Print info string with algorithm details

## Linear Solver Options
- `linear_solver::String = "mumps"`: Linear solver to use (mumps, ma27, ma57, ma86, ma97, pardiso, wsmp, etc.)
- `linear_system_scaling::String = "none"`: Method for scaling linear system (none, mc19, slack-based)
- `hsllib::String = ""`: Path to HSL library (if using HSL solvers)
- `pardisolib::String = ""`: Path to Pardiso library (if using Pardiso)
- `linear_scaling_on_demand::String = "yes"`: Enable scaling on demand for linear systems

## NLP Scaling Options
- `nlp_scaling_method::String = "gradient-based"`: Scaling method for NLP (none, user-scaling, gradient-based, equilibration-based)
- `nlp_scaling_max_gradient::Float64 = 100.0`: Maximum gradient after scaling
- `honor_original_bounds::String = "no"`: Honor original variable bounds after scaling
- `check_derivatives_for_naninf::String = "no"`: Check derivatives for NaN/Inf values

## Barrier Parameter Options
- `mu_strategy::String = "monotone"`: Update strategy for barrier parameter (monotone, adaptive)
- `mu_oracle::String = "quality-function"`: Oracle for adaptive mu strategy
- `mu_init::Float64 = 0.1`: Initial value for barrier parameter
- `adaptive_mu_globalization::String = "obj-constr-filter"`: Globalization strategy for adaptive mu

## Warm Start Options
- `warm_start_init_point::String = "no"`: Use warm start from previous solution

## Hessian Options
- `hessian_approximation::String = "exact"`: How to approximate the Hessian (exact, limited-memory)
- `limited_memory_max_history::Int = 6`: History size for limited-memory Hessian approximation
- `limited_memory_update_type::String = "bfgs"`: Quasi-Newton update formula for limited-memory approximation (bfgs, sr1)

## Line Search Options
- `accept_every_trial_step::String = "no"`: Accept every trial step (disables line search)
- `line_search_method::String = "filter"`: Line search method (filter, penalty, cg-penalty)

## Restoration Phase Options
- `expect_infeasible_problem::String = "no"`: Enable if problem is expected to be infeasible

## Additional Options
- `additional_options::Dict{String, Any} = Dict()`: Dictionary to set any other Ipopt option not explicitly listed above.
  See https://coin-or.github.io/Ipopt/OPTIONS.html for the full list of available options.

# Examples

```julia
using OptimizationBase, OptimizationIpopt

# Basic usage with default settings
opt = IpoptOptimizer()

# Customized settings
opt = IpoptOptimizer(
    linear_solver = "ma57", # needs HSL solvers configured
    nlp_scaling_method = "equilibration-based",
    hessian_approximation = "limited-memory",
    additional_options = Dict(
        "alpha_for_y" => "primal",
        "recalc_y" => "yes"
    )
)

# Solve with common interface arguments
result = solve(prob, opt;
    reltol = 1e-8,      # Sets Ipopt's tol
    maxiters = 5000,    # Sets Ipopt's max_iter
    maxtime = 300.0,    # Sets Ipopt's max_wall_time (in seconds)
    verbose = 3         # Sets Ipopt's print_level
)
```

# References

For complete documentation of all Ipopt options, see:
https://coin-or.github.io/Ipopt/OPTIONS.html
"""
@kwdef struct IpoptOptimizer
    # Most common Ipopt-specific options (excluding common interface options)

    # Termination
    acceptable_tol::Float64 = 1.0e-6
    acceptable_iter::Int = 15
    dual_inf_tol::Float64 = 1.0
    constr_viol_tol::Float64 = 1.0e-4
    compl_inf_tol::Float64 = 1.0e-4

    # Output options
    print_timing_statistics::String = "no"
    print_info_string::String = "no"

    # Linear solver
    linear_solver::String = "mumps"
    linear_system_scaling::String = "none"
    hsllib::String = ""
    pardisolib::String = ""
    linear_scaling_on_demand = "yes"

    # NLP options
    nlp_scaling_method::String = "gradient-based"
    nlp_scaling_max_gradient::Float64 = 100.0
    honor_original_bounds::String = "no"
    check_derivatives_for_naninf::String = "no"

    # Barrier parameter
    mu_strategy::String = "monotone"
    mu_oracle::String = "quality-function"
    mu_init::Float64 = 0.1
    adaptive_mu_globalization::String = "obj-constr-filter"

    # Warm start
    warm_start_init_point::String = "no"

    # Hessian approximation
    hessian_approximation::String = "exact"
    limited_memory_max_history::Int = 6
    limited_memory_update_type::String = "bfgs"

    # Line search
    accept_every_trial_step::String = "no"
    line_search_method::String = "filter"

    # Restoration phase
    expect_infeasible_problem::String = "no"

    # Additional options for any other Ipopt parameters
    additional_options::Dict{String, Any} = Dict{String, Any}()
end

function SciMLBase.has_init(alg::IpoptOptimizer)
    return true
end

SciMLBase.allowscallback(alg::IpoptOptimizer) = true

# Compatibility with OptimizationBase@v3
function SciMLBase.supports_opt_cache_interface(alg::IpoptOptimizer)
    return true
end

function SciMLBase.requiresgradient(opt::IpoptOptimizer)
    return true
end
function SciMLBase.requireshessian(opt::IpoptOptimizer)
    return true
end
function SciMLBase.requiresconsjac(opt::IpoptOptimizer)
    return true
end
function SciMLBase.requiresconshess(opt::IpoptOptimizer)
    return true
end

function SciMLBase.allowsbounds(opt::IpoptOptimizer)
    return true
end
function SciMLBase.allowsconstraints(opt::IpoptOptimizer)
    return true
end

include("cache.jl")
include("callback.jl")

function __map_optimizer_args(
        cache,
        opt::IpoptOptimizer;
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        verbose = false,
        progress::Bool = false,
        callback = nothing
    )
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

    # Set up progress callback
    progress_callback = IpoptProgressLogger(
        progress, callback, prob, cache.n, cache.num_cons, maxiters, cache.iterations
    )
    intermediate = (args...) -> progress_callback(args...)
    Ipopt.SetIntermediateCallback(prob, intermediate)

    # Apply all options from struct using reflection and type dispatch
    for field in propertynames(opt)
        field == :additional_options && continue  # Skip the dict field

        field_str = string(field)
        value = getproperty(opt, field)

        # Apply option based on type
        if value isa Int
            Ipopt.AddIpoptIntOption(prob, field_str, value)
        elseif value isa Float64
            Ipopt.AddIpoptNumOption(prob, field_str, value)
        elseif value isa String
            Ipopt.AddIpoptStrOption(prob, field_str, value)
        end
    end

    # Apply additional options with type dispatch
    for (key, value) in opt.additional_options
        if value isa Int
            Ipopt.AddIpoptIntOption(prob, key, value)
        elseif value isa Float64
            Ipopt.AddIpoptNumOption(prob, key, float(value))
        elseif value isa String
            Ipopt.AddIpoptStrOption(prob, key, value)
        else
            error("Unsupported option type $(typeof(value)) for option $key. Must be Int, Float64, or String")
        end
    end

    # Override with common interface arguments if provided
    optkeys = keys(opt.additional_options)
    !isnothing(reltol) && !in("tol", optkeys) && Ipopt.AddIpoptNumOption(prob, "tol", reltol)
    !isnothing(maxiters) && !in("max_iter", optkeys) && Ipopt.AddIpoptIntOption(prob, "max_iter", maxiters)
    !isnothing(maxtime) && !in("max_wall_time", optkeys) && Ipopt.AddIpoptNumOption(prob, "max_wall_time", Float64(maxtime))

    # Set Ipopt print_level
    if !in("print_level", optkeys)
        # Priority: verbose kwarg (backward compatibility) > ipopt_verbosity toggle
        if verbose isa Bool
            Ipopt.AddIpoptIntOption(prob, "print_level", verbose * 5)
        elseif verbose isa Int
            Ipopt.AddIpoptIntOption(prob, "print_level", verbose)
        else
            # verbose is an OptimizationVerbosity object
            print_level = SciMLLogging.verbosity_to_int(verbose.ipopt_verbosity)
            Ipopt.AddIpoptIntOption(prob, "print_level", print_level)
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
            Ipopt.Feasible_Point_Found,
        ]
        return ReturnCode.Success
    elseif status in [
            Ipopt.Infeasible_Problem_Detected,
            Ipopt.Search_Direction_Becomes_Too_Small,
            Ipopt.Diverging_Iterates,
        ]
        return ReturnCode.Infeasible
    elseif status == Ipopt.Maximum_Iterations_Exceeded
        return ReturnCode.MaxIters
    elseif status in [
            Ipopt.Maximum_CpuTime_Exceeded
            Ipopt.Maximum_WallTime_Exceeded
        ]
        return ReturnCode.MaxTime
    else
        return ReturnCode.Failure
    end
end

function SciMLBase.__solve(cache::IpoptCache)
    maxiters = OptimizationBase._check_and_convert_maxiters(cache.solver_args.maxiters)
    maxtime = OptimizationBase._check_and_convert_maxtime(cache.solver_args.maxtime)

    opt_setup = __map_optimizer_args(
        cache,
        cache.opt;
        abstol = cache.solver_args.abstol,
        reltol = cache.solver_args.reltol,
        maxiters = maxiters,
        maxtime = maxtime,
        verbose = cache.solver_args.verbose,
        progress = cache.progress,
        callback = cache.callback
    )

    opt_setup.x .= cache.reinit_cache.u0

    start_time = time()
    status = Ipopt.IpoptSolve(opt_setup)

    opt_ret = map_retcode(status)

    if cache.progress
        # Set progressbar to 1 to finish it
        Base.@logmsg(Base.LogLevel(-1), "", progress = 1, _id = :OptimizationIpopt)
    end

    minimum = opt_setup.obj_val
    minimizer = opt_setup.x

    stats = OptimizationBase.OptimizationStats(;
        time = time() - start_time,
        iterations = cache.iterations[], fevals = cache.f_calls, gevals = cache.f_grad_calls
    )

    finalize(opt_setup)

    return SciMLBase.build_solution(
        cache,
        cache.opt,
        minimizer,
        minimum;
        original = opt_setup,
        retcode = opt_ret,
        stats = stats
    )
end

function SciMLBase.__init(
        prob::OptimizationProblem,
        opt::IpoptOptimizer;
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        progress::Bool = false,
        kwargs...
    )
    cache = IpoptCache(
        prob, opt;
        maxiters,
        maxtime,
        abstol,
        reltol,
        progress,
        kwargs...
    )
    cache.reinit_cache.u0 .= prob.u0

    return cache
end

end # OptimizationIpopt
