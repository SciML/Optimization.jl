#This file lets you drive SciPy optimizers through SciML's Optimization.jl API.
module OptimizationSciPy

using Reexport
@reexport using Optimization
using Optimization.SciMLBase
using PythonCall

# We keep a handle to the actual Python SciPy module here.
const scipy = PythonCall.pynew()

function __init__()
    PythonCall.pycopy!(scipy, pyimport("scipy"))
end

# Make sure whatever we got back is a plain Julia Vector{T}.
function ensure_julia_array(x, ::Type{T}=Float64) where T
    x isa Vector{T} && return x
    return convert(Vector{T}, x isa Py ? pyconvert(Vector, x) : x)
end

# Pull a human-readable message out of the SciPy result object.
function safe_get_message(result)
    pyhasattr(result, "message") || return "Optimization completed"
    msg = result.message
    if pyisinstance(msg, pybuiltins.str)
        return pyconvert(String, msg)
    end
    if pyisinstance(msg, pybuiltins.list) || pyisinstance(msg, pybuiltins.tuple)
        return join(pyconvert(Vector{String}, msg), ", ")
    end
    return string(pytypeof(msg))
end

# Squash any kind of numeric object down to a Julia Float64.
function safe_to_float(x)
    x isa Float64 && return x
    x isa Number  && return Float64(x)

    if x isa Py
        if pyhasattr(x, "item")
            v = pyconvert(Float64, x.item(), nothing)
            v !== nothing && return v
        end
        v = pyconvert(Float64, x, nothing)
        v !== nothing && return v
    end

    error("Cannot convert object to Float64: $(typeof(x))")
end

# Gather timing / iteration counts and wrap them in OptimizationStats.
function extract_stats(result, time_elapsed)
    stats_dict = Dict{Symbol, Any}(
        :iterations => 0,
        :time => time_elapsed,
        :fevals => 0,
        :gevals => 0,
        :hevals => 0
    )
    if pyhasattr(result, "nit") && !pyis(result.nit, pybuiltins.None)
        stats_dict[:iterations] = pyconvert(Int, result.nit)
    end
    if pyhasattr(result, "nfev") && !pyis(result.nfev, pybuiltins.None)
        stats_dict[:fevals] = pyconvert(Int, result.nfev)
    end
    if pyhasattr(result, "njev") && !pyis(result.njev, pybuiltins.None)
        stats_dict[:gevals] = pyconvert(Int, result.njev)
    elseif pyhasattr(result, "ngrad") && !pyis(result.ngrad, pybuiltins.None)
        stats_dict[:gevals] = pyconvert(Int, result.ngrad)
    end
    if pyhasattr(result, "nhev") && !pyis(result.nhev, pybuiltins.None)
        stats_dict[:hevals] = pyconvert(Int, result.nhev)
    end
    return Optimization.OptimizationStats(; stats_dict...)
end

# Map SciPy status integers onto SciML ReturnCode symbols.
function scipy_status_to_retcode(status::Int, success::Bool)
    if success
        return SciMLBase.ReturnCode.Success
    end
    return if status == 0
        SciMLBase.ReturnCode.Success
    elseif status == 1
        SciMLBase.ReturnCode.MaxIters
    elseif status == 2
        SciMLBase.ReturnCode.Infeasible
    elseif status == 3
        SciMLBase.ReturnCode.Unstable
    elseif status == 4
        SciMLBase.ReturnCode.Terminated
    elseif status == 9  
        SciMLBase.ReturnCode.MaxIters
    else
        SciMLBase.ReturnCode.Failure
    end
end

# Tiny structs that tag which SciPy algorithm the user picked.
abstract type ScipyOptimizer end

struct ScipyMinimize <: ScipyOptimizer
    method::String
    function ScipyMinimize(method::String)
        valid_methods = ["Nelder-Mead", "Powell", "CG", "BFGS", "Newton-CG",
                        "L-BFGS-B", "TNC", "COBYLA", "COBYQA", "SLSQP",
                        "trust-constr", "dogleg", "trust-ncg", "trust-krylov",
                        "trust-exact"]
        if !(method in valid_methods)
            throw(ArgumentError("Invalid method: $method. Valid methods are: $(join(valid_methods, ", "))"))
        end
        new(method)
    end
end
ScipyMinimize() = ScipyMinimize("BFGS")

ScipyNelderMead() = ScipyMinimize("Nelder-Mead")
ScipyPowell() = ScipyMinimize("Powell")
ScipyCG() = ScipyMinimize("CG")
ScipyBFGS() = ScipyMinimize("BFGS")
ScipyNewtonCG() = ScipyMinimize("Newton-CG")
ScipyLBFGSB() = ScipyMinimize("L-BFGS-B")
ScipyTNC() = ScipyMinimize("TNC")
ScipyCOBYLA() = ScipyMinimize("COBYLA")
ScipyCOBYQA() = ScipyMinimize("COBYQA")
ScipySLSQP() = ScipyMinimize("SLSQP")
ScipyTrustConstr() = ScipyMinimize("trust-constr")
ScipyDogleg() = ScipyMinimize("dogleg")
ScipyTrustNCG() = ScipyMinimize("trust-ncg")
ScipyTrustKrylov() = ScipyMinimize("trust-krylov")
ScipyTrustExact() = ScipyMinimize("trust-exact")

struct ScipyMinimizeScalar <: ScipyOptimizer
    method::String
    function ScipyMinimizeScalar(method::String="brent")
        valid_methods = ["brent", "bounded", "golden"]
        if !(method in valid_methods)
            throw(ArgumentError("Invalid method: $method. Valid methods are: $(join(valid_methods, ", "))"))
        end
        new(method)
    end
end

ScipyBrent() = ScipyMinimizeScalar("brent")
ScipyBounded() = ScipyMinimizeScalar("bounded")
ScipyGolden() = ScipyMinimizeScalar("golden")

struct ScipyLeastSquares <: ScipyOptimizer
    method::String
    loss::String
    function ScipyLeastSquares(; method::String="trf", loss::String="linear")
        valid_methods = ["trf", "dogbox", "lm"]
        valid_losses = ["linear", "soft_l1", "huber", "cauchy", "arctan"]
        if !(method in valid_methods)
            throw(ArgumentError("Invalid method: $method. Valid methods are: $(join(valid_methods, ", "))"))
        end
        if !(loss in valid_losses)
            throw(ArgumentError("Invalid loss: $loss. Valid loss functions are: $(join(valid_losses, ", "))"))
        end
        new(method, loss)
    end
end

ScipyLeastSquaresTRF() = ScipyLeastSquares(method="trf")
ScipyLeastSquaresDogbox() = ScipyLeastSquares(method="dogbox")
ScipyLeastSquaresLM() = ScipyLeastSquares(method="lm")

struct ScipyRootScalar <: ScipyOptimizer
    method::String
    function ScipyRootScalar(method::String="brentq")
        valid_methods = ["brentq", "brenth", "bisect", "ridder", "newton", "secant", "halley", "toms748"]
        if !(method in valid_methods)
            throw(ArgumentError("Invalid method: $method. Valid methods are: $(join(valid_methods, ", "))"))
        end
        new(method)
    end
end

struct ScipyRoot <: ScipyOptimizer
    method::String
    function ScipyRoot(method::String="hybr")
        valid_methods = ["hybr", "lm", "broyden1", "broyden2", "anderson",
                        "linearmixing", "diagbroyden", "excitingmixing", 
                        "krylov", "df-sane"]
        if !(method in valid_methods)
            throw(ArgumentError("Invalid method: $method. Valid methods are: $(join(valid_methods, ", "))"))
        end
        new(method)
    end
end

struct ScipyLinprog <: ScipyOptimizer
    method::String
    function ScipyLinprog(method::String="highs")
        valid_methods = ["highs", "highs-ds", "highs-ipm", "interior-point", 
                        "revised simplex", "simplex"]
        if !(method in valid_methods)
            throw(ArgumentError("Invalid method: $method. Valid methods are: $(join(valid_methods, ", "))"))
        end
        new(method)
    end
end

struct ScipyMilp <: ScipyOptimizer end
struct ScipyDifferentialEvolution <: ScipyOptimizer end
struct ScipyBasinhopping <: ScipyOptimizer end
struct ScipyDualAnnealing <: ScipyOptimizer end
struct ScipyShgo <: ScipyOptimizer end
struct ScipyDirect <: ScipyOptimizer end
struct ScipyBrute <: ScipyOptimizer end

for opt_type in [:ScipyMinimize, :ScipyDifferentialEvolution, :ScipyBasinhopping, 
                  :ScipyDualAnnealing, :ScipyShgo, :ScipyDirect, :ScipyBrute,
                  :ScipyLinprog, :ScipyMilp]
    @eval begin
        SciMLBase.allowsbounds(::$opt_type) = true
        SciMLBase.supports_opt_cache_interface(::$opt_type) = true
    end
end

for opt_type in [:ScipyMinimizeScalar, :ScipyRootScalar, :ScipyLeastSquares]
    @eval begin
        SciMLBase.supports_opt_cache_interface(::$opt_type) = true
    end
end

SciMLBase.supports_opt_cache_interface(::ScipyRoot) = true

function SciMLBase.requiresgradient(opt::ScipyMinimize)
    gradient_free = ["Nelder-Mead", "Powell", "COBYLA", "COBYQA"]
    return !(opt.method in gradient_free)
end

for opt_type in [:ScipyDifferentialEvolution, :ScipyBasinhopping, 
                  :ScipyDualAnnealing, :ScipyShgo, :ScipyDirect, :ScipyBrute,
                  :ScipyMinimizeScalar, :ScipyLeastSquares, :ScipyRootScalar,
                  :ScipyRoot, :ScipyLinprog, :ScipyMilp]
    @eval SciMLBase.requiresgradient(::$opt_type) = false
end

function SciMLBase.requireshessian(opt::ScipyMinimize)
    hessian_methods = ["Newton-CG", "dogleg", "trust-ncg", "trust-exact", "trust-krylov"]
    return opt.method in hessian_methods
end

function SciMLBase.requireshessian(opt::ScipyRootScalar)
    return opt.method == "halley"
end

function SciMLBase.allowsconstraints(opt::ScipyMinimize)
    return opt.method in ["SLSQP", "trust-constr", "COBYLA", "COBYQA"]
end

function SciMLBase.requiresconsjac(opt::ScipyMinimize)
    return opt.method in ["SLSQP", "trust-constr"]
end

SciMLBase.allowsconstraints(::ScipyShgo) = true
SciMLBase.allowsconstraints(::ScipyLinprog) = true
SciMLBase.allowsconstraints(::ScipyMilp) = true

function SciMLBase.allowsbounds(opt::ScipyMinimizeScalar)
    return opt.method == "bounded"
end

function SciMLBase.allowsbounds(opt::ScipyLeastSquares)
    return opt.method in ["trf", "dogbox"]
end

function SciMLBase.allowsbounds(opt::ScipyRootScalar)
    return opt.method in ["brentq", "brenth", "bisect", "ridder"]
end

SciMLBase.allowsbounds(::ScipyRoot) = false

function SciMLBase.__init(prob::SciMLBase.OptimizationProblem, opt::ScipyOptimizer;
                         cons_tol = 1e-6,
                         callback = (args...) -> (false),
                         progress = false, 
                         kwargs...)
    requires_bounds = opt isa Union{ScipyDifferentialEvolution, ScipyDirect, ScipyDualAnnealing, ScipyBrute}
    if requires_bounds && (isnothing(prob.lb) || isnothing(prob.ub))
        throw(SciMLBase.IncompatibleOptimizerError("$(typeof(opt)) requires bounds"))
    end
    if opt isa ScipyMinimizeScalar && length(prob.u0) != 1
        throw(ArgumentError("ScipyMinimizeScalar requires exactly 1 variable, got $(length(prob.u0)). Use ScipyMinimize for multivariate problems."))
    end
    if opt isa ScipyRootScalar && length(prob.u0) != 1
        throw(ArgumentError("ScipyRootScalar requires exactly 1 variable, got $(length(prob.u0)). Use ScipyRoot for multivariate problems."))
    end
    if opt isa ScipyMinimizeScalar && opt.method == "bounded"
        if isnothing(prob.lb) || isnothing(prob.ub)
            throw(ArgumentError("ScipyMinimizeScalar with method='bounded' requires bounds"))
        end
    end
    if opt isa ScipyRootScalar && opt.method in ["brentq", "brenth", "bisect", "ridder"]
        if isnothing(prob.lb) || isnothing(prob.ub)
            throw(ArgumentError("ScipyRootScalar with method='$(opt.method)' requires bracket (bounds)"))
        end
    end
    if !isnothing(prob.lb) && !isnothing(prob.ub)
        @assert length(prob.lb) == length(prob.ub) "Bounds must have the same length"
        @assert all(prob.lb .<= prob.ub) "Lower bounds must be less than or equal to upper bounds"
    end
    return OptimizationCache(prob, opt; cons_tol, callback, progress, kwargs...)
end

function SciMLBase.__solve(cache::OptimizationCache{F,RC,LB,UB,LC,UC,S,O,D,P,C}) where 
                          {F,RC,LB,UB,LC,UC,S,O<:ScipyMinimize,D,P,C}
    local cons_cache = nothing
    if !isnothing(cache.f.cons) && !isnothing(cache.lcons)
        cons_cache = zeros(eltype(cache.u0), length(cache.lcons))
    end
    _loss = _create_loss(cache)
    maxiters = Optimization._check_and_convert_maxiters(cache.solver_args.maxiters)
    abstol = cache.solver_args.abstol
    reltol = cache.solver_args.reltol
    options = Dict{String, Any}()
    if cache.opt.method == "trust-constr"
        options["initial_tr_radius"] = 1.0
        options["verbose"] = 0
        options["finite_diff_rel_step"] = 1e-8
        options["gtol"] = 1e-10
        options["maxiter"] = 50000
    elseif cache.opt.method in ["dogleg", "trust-ncg", "trust-krylov", "trust-exact"]
        options["gtol"] = 1e-10
        options["maxiter"] = 50000
    end
    if !isnothing(maxiters)
        options["maxiter"] = maxiters
    end
    if !isnothing(abstol)
        if cache.opt.method in ["Nelder-Mead", "Powell"]
            options["xatol"] = abstol
        elseif cache.opt.method in ["L-BFGS-B", "TNC", "SLSQP", "trust-constr"]
            options["ftol"] = abstol
        elseif cache.opt.method == "COBYQA"
            options["feasibility_tol"] = abstol
        end
    end
    if !isnothing(reltol)
        if cache.opt.method in ["CG", "BFGS", "Newton-CG", "L-BFGS-B", "TNC", "SLSQP", "trust-constr"]
            options["gtol"] = reltol
        end
    end
    _merge_solver_kwargs!(options, cache.solver_args)
    jac = nothing
    if SciMLBase.requiresgradient(cache.opt) && !isnothing(cache.f.grad)
        _grad = function(θ)
            θ_julia = ensure_julia_array(θ, eltype(cache.u0))
            grad = zeros(eltype(cache.u0), length(θ_julia))
            cache.f.grad(grad, θ_julia, cache.p)
            return cache.sense === Optimization.MaxSense ? -grad : grad
        end
        jac = _grad
    end
    hess = nothing
    if SciMLBase.requireshessian(cache.opt)
        if !isnothing(cache.f.hess)
            _hess = function(θ)
                θ_julia = ensure_julia_array(θ, eltype(cache.u0))
                H = zeros(eltype(cache.u0), length(θ_julia), length(θ_julia))
                cache.f.hess(H, θ_julia, cache.p)
                return cache.sense === Optimization.MaxSense ? -H : H
            end
            hess = _hess
        else
            if cache.opt.method in ["trust-constr", "dogleg", "trust-ncg", "trust-krylov", "trust-exact"]
                options["hess"] = "BFGS"
            else
                throw(ArgumentError("Method $(cache.opt.method) requires Hessian but none was provided"))
            end
        end
    end
    bounds = nothing
    if !isnothing(cache.lb) && !isnothing(cache.ub)
        if cache.opt.method in ["L-BFGS-B", "TNC", "SLSQP", "trust-constr", "COBYLA", "COBYQA"]
            bounds = scipy.optimize.Bounds(cache.lb, cache.ub)
        end
    end
    constraints = pylist([])
    if SciMLBase.allowsconstraints(cache.opt)
        if !isnothing(cache.f.cons) && !isnothing(cons_cache)
            lcons = cache.lcons
            ucons = cache.ucons
            _cons_func = function(θ)
                θ_julia = ensure_julia_array(θ, eltype(cache.u0))
                cons_cache .= zero(eltype(cons_cache))
                if hasmethod(cache.f.cons, Tuple{typeof(cons_cache), typeof(θ_julia), typeof(cache.p)})
                    cache.f.cons(cons_cache, θ_julia, cache.p)
                else
                    cache.f.cons(cons_cache, θ_julia)
                end
                return cons_cache
            end
            cons_jac = "2-point"
            if SciMLBase.requiresconsjac(cache.opt) && !isnothing(cache.f.cons_j)
                cons_j_cache = zeros(eltype(cache.u0), length(lcons), length(cache.u0))
                _cons_jac = function(θ)
                    θ_julia = ensure_julia_array(θ, eltype(cache.u0))
                    if hasmethod(cache.f.cons_j, Tuple{typeof(cons_j_cache), typeof(θ_julia), typeof(cache.p)})
                        cache.f.cons_j(cons_j_cache, θ_julia, cache.p)
                    else
                        cache.f.cons_j(cons_j_cache, θ_julia)
                    end
                    return cons_j_cache
                end
                cons_jac = _cons_jac
            end
            # user-controlled NonlinearConstraint extras
            keep_feasible_flag = get(cache.solver_args, :keep_feasible, false)
            jac_sparsity      = get(cache.solver_args, :jac_sparsity, nothing)
            nlc = scipy.optimize.NonlinearConstraint(
                _cons_func,
                lcons,
                ucons;
                jac                    = cons_jac,
                keep_feasible          = keep_feasible_flag,
                finite_diff_rel_step   = get(cache.solver_args, :cons_tol, 1e-8),
                finite_diff_jac_sparsity = jac_sparsity,
            )
            constraints = pylist([nlc])
        end
    elseif !isnothing(cache.f.cons)
        throw(ArgumentError("Method $(cache.opt.method) does not support constraints. Use SLSQP, trust-constr, COBYLA, or COBYQA instead."))
    end
    # allow users to specify a Hessian update strategy (e.g. "BFGS", "SR1")
    if cache.opt.method == "trust-constr"
        hess_update = get(cache.solver_args, :hess_update, nothing)
        if hess_update !== nothing
            hess = hess_update
        end
    end
    t0 = time()
    result = nothing
    try
        result = scipy.optimize.minimize(
            _loss,
            cache.u0,
            method = cache.opt.method,
            jac = jac,
            hess = hess,
            bounds = bounds,
            constraints = constraints,
            options = pydict(options)
        )
    catch e
        if e isa PythonCall.Core.PyException
            py_msg = sprint(showerror, e)
            if occursin("Optimization halted by callback", py_msg)
                throw(ErrorException("Optimization halted by callback"))
            elseif occursin("Optimization halted: time limit exceeded", py_msg)
                throw(ErrorException("Optimization halted: time limit exceeded"))
            else
                throw(ErrorException("SciPy optimization failed: $py_msg"))
            end
        else
            rethrow(e)
        end
    end
    if isnothing(result)
        throw(ErrorException("Optimization failed to return a result"))
    end
    t1 = time()
    if pyis(result.x, pybuiltins.None)
        minimizer = fill(NaN, length(cache.u0))
    else
        minimizer = pyconvert(Vector{eltype(cache.u0)}, result.x)
    end
    minimum = pyis(result.fun, pybuiltins.None) ? NaN : safe_to_float(result.fun)
    py_success = pyconvert(Bool, pybool(result.success))
    py_message = safe_get_message(result)
    status = 0
    if pyhasattr(result, "status")
        try
            status = pyconvert(Int, result.status)
        catch
        end
    end
    if cache.sense === Optimization.MaxSense
        minimum = -minimum
    end
    retcode = scipy_status_to_retcode(status, py_success)
    if retcode != SciMLBase.ReturnCode.Success
        @debug "ScipyMinimize convergence: $(py_message)"
    end
    stats = extract_stats(result, t1 - t0)
    return SciMLBase.build_solution(cache, cache.opt, minimizer, minimum;
                                   original = result,
                                   retcode = retcode,
                                   stats = stats)
end

function SciMLBase.__solve(cache::OptimizationCache{F,RC,LB,UB,LC,UC,S,O,D,P,C}) where 
                          {F,RC,LB,UB,LC,UC,S,O<:ScipyMinimizeScalar,D,P,C}
    maxtime = get(cache.solver_args, :maxtime, nothing)
    start_time = time()
    _loss = function(θ)
        if !isnothing(maxtime) && (time() - start_time) > maxtime
            error("Optimization halted: time limit exceeded")
        end
        θ_vec = [θ]
        x = cache.f(θ_vec, cache.p)
        x = isa(x, Tuple) ? x : (x,)
        opt_state = Optimization.OptimizationState(u = θ_vec, objective = x[1])
        if cache.callback(opt_state, x...)
            error("Optimization halted by callback")
        end
        return x[1]
    end
    kwargs = Dict{Symbol, Any}()
    if cache.opt.method == "bounded"
        if !isnothing(cache.lb) && !isnothing(cache.ub)
            kwargs[:bounds] = (cache.lb[1], cache.ub[1])
        else
            throw(ArgumentError("Bounded method requires bounds"))
        end
    end
    _merge_solver_kwargs!(kwargs, cache.solver_args)
    t0 = time()
    result = nothing
    try
        result = scipy.optimize.minimize_scalar(
            _loss,
            method = cache.opt.method;
            kwargs...
        )
    catch e
        if e isa PythonCall.Core.PyException
            py_msg = sprint(showerror, e)
            if occursin("Optimization halted", py_msg)
                throw(ErrorException(py_msg))
            else
                throw(ErrorException("SciPy optimization failed: $py_msg"))
            end
        else
            rethrow(e)
        end
    end
    if isnothing(result)
        throw(ErrorException("Optimization failed to return a result"))
    end
    t1 = time()
    if pyis(result.x, pybuiltins.None)
        minimizer = [NaN]
    else
        minimizer = [safe_to_float(result.x)]
    end
    minimum = pyis(result.fun, pybuiltins.None) ? NaN : safe_to_float(result.fun)
    py_success = pyconvert(Bool, pybool(result.success))
    if cache.sense === Optimization.MaxSense
        minimum = -minimum
    end
    retcode = py_success ? SciMLBase.ReturnCode.Success : SciMLBase.ReturnCode.Failure
    stats = extract_stats(result, t1 - t0)
    return SciMLBase.build_solution(cache, cache.opt, minimizer, minimum;
                                   original = result,
                                   retcode = retcode,
                                   stats = stats)
end

function SciMLBase.__solve(cache::OptimizationCache{F,RC,LB,UB,LC,UC,S,O,D,P,C}) where 
                          {F,RC,LB,UB,LC,UC,S,O<:ScipyLeastSquares,D,P,C}
    _residuals = nothing
    if hasfield(typeof(cache.f), :f) && (cache.f.f isa ResidualObjective)
        real_res = (cache.f.f)::ResidualObjective
        _residuals = function(θ)
            θ_julia = ensure_julia_array(θ, eltype(cache.u0))
            return real_res.residual(θ_julia, cache.p)
        end
    else
        _residuals = _create_loss(cache; vector_output=true)
    end
    kwargs = Dict{Symbol, Any}()
    kwargs[:method] = cache.opt.method
    kwargs[:loss] = cache.opt.loss
    if !isnothing(cache.lb) && !isnothing(cache.ub) && cache.opt.method in ["trf", "dogbox"]
        kwargs[:bounds] = (cache.lb, cache.ub)
    elseif cache.opt.method == "lm" && (!isnothing(cache.lb) || !isnothing(cache.ub))
        @warn "Method 'lm' does not support bounds. Ignoring bounds."
    end
    kwargs[:jac] = "2-point"
    maxiters = Optimization._check_and_convert_maxiters(cache.solver_args.maxiters)
    if !isnothing(maxiters)
        kwargs[:max_nfev] = maxiters
    end
    if !isnothing(cache.solver_args.abstol)
        kwargs[:ftol] = cache.solver_args.abstol
    end
    if !isnothing(cache.solver_args.reltol)
        kwargs[:gtol] = cache.solver_args.reltol
    end
    _merge_solver_kwargs!(kwargs, cache.solver_args)
    t0 = time()
    result = nothing
    try
        result = scipy.optimize.least_squares(
            _residuals,
            cache.u0;
            kwargs...
        )
    catch e
        if e isa PythonCall.Core.PyException
            py_msg = sprint(showerror, e)
            if occursin("Optimization halted by callback", py_msg)
                throw(ErrorException("Optimization halted by callback"))
            elseif occursin("Optimization halted: time limit exceeded", py_msg)
                throw(ErrorException("Optimization halted: time limit exceeded"))
            else
                throw(ErrorException("SciPy optimization failed: $py_msg"))
            end
        else
            rethrow(e)
        end
    end
    if isnothing(result)
        throw(ErrorException("Optimization failed to return a result"))
    end
    t1 = time()
    if pyis(result.x, pybuiltins.None)
        minimizer = fill(NaN, length(cache.u0))
    else
        minimizer = pyconvert(Vector{eltype(cache.u0)}, result.x)
    end
    minimum = safe_to_float(result.cost)
    py_success = pyconvert(Bool, pybool(result.success))
    py_message = safe_get_message(result)
    status = 0
    if pyhasattr(result, "status")
        try
            status = pyconvert(Int, result.status)
        catch
        end
    end
    retcode = scipy_status_to_retcode(status, py_success)
    if retcode != SciMLBase.ReturnCode.Success
        @debug "ScipyLeastSquares convergence: $(py_message)"
    end
    stats = extract_stats(result, t1 - t0)
    return SciMLBase.build_solution(cache, cache.opt, minimizer, minimum;
                                   original = result,
                                   retcode = retcode,
                                   stats = stats)
end

function SciMLBase.__solve(cache::OptimizationCache{F,RC,LB,UB,LC,UC,S,O,D,P,C}) where 
                          {F,RC,LB,UB,LC,UC,S,O<:ScipyRootScalar,D,P,C}
    x0 = cache.u0[1]
    maxtime = get(cache.solver_args, :maxtime, nothing)
    start_time = time()
    _func = function(θ)
        if !isnothing(maxtime) && (time() - start_time) > maxtime
            error("Optimization halted: time limit exceeded")
        end
        θ_vec = [θ]
        x = cache.f(θ_vec, cache.p)
        x = isa(x, Tuple) ? x : (x,)
        opt_state = Optimization.OptimizationState(u = θ_vec, objective = x[1])
        if cache.callback(opt_state, x...)
            error("Optimization halted by callback")
        end
        return x[1]
    end
    kwargs = Dict{Symbol, Any}()
    bracketing_methods = ["brentq", "brenth", "bisect", "ridder"]
    is_bracketing = cache.opt.method in bracketing_methods
    if is_bracketing
        if !isnothing(cache.lb) && !isnothing(cache.ub)
            kwargs[:bracket] = pytuple([cache.lb[1], cache.ub[1]])
        else
            throw(ArgumentError("Method $(cache.opt.method) requires bracket (bounds)"))
        end
    else
        kwargs[:x0] = x0
    end
    if cache.opt.method == "newton" && !isnothing(cache.f.grad)
        _fprime = function(θ)
            grad = zeros(eltype(cache.u0), 1)
            cache.f.grad(grad, [θ], cache.p)
            return grad[1]
        end
        kwargs[:fprime] = _fprime
    elseif cache.opt.method == "halley"
        if !isnothing(cache.f.grad) && !isnothing(cache.f.hess)
            _fprime = function(θ)
                grad = zeros(eltype(cache.u0), 1)
                cache.f.grad(grad, [θ], cache.p)
                return grad[1]
            end
            _fprime2 = function(θ)
                hess = zeros(eltype(cache.u0), 1, 1)
                cache.f.hess(hess, [θ], cache.p)
                return hess[1, 1]
            end
            kwargs[:fprime] = _fprime
            kwargs[:fprime2] = _fprime2
        else
            throw(ArgumentError("Method 'halley' requires both gradient and Hessian"))
        end
    end
    _merge_solver_kwargs!(kwargs, cache.solver_args)
    t0 = time()
    result = nothing
    try
        result = scipy.optimize.root_scalar(
            _func;
            method = cache.opt.method,
            kwargs...
        )
    catch e
        if e isa PythonCall.Core.PyException
            py_msg = sprint(showerror, e)
            if occursin("Optimization halted", py_msg)
                throw(ErrorException(py_msg))
            else
                throw(ErrorException("SciPy root finding failed: $py_msg"))
            end
        else
            rethrow(e)
        end
    end
    if isnothing(result)
        throw(ErrorException("Root finding failed to return a result"))
    end
    t1 = time()
    if pyis(result.root, pybuiltins.None)
        minimizer = [NaN]
        root_julia = NaN
        minimum = NaN
    else
        val = safe_to_float(result.root)
        minimizer = [val]
        root_julia = val
        minimum = abs(_func(root_julia))
    end
    converged = pyhasattr(result, "converged") ? pyconvert(Bool, pybool(result.converged)) : abs(_func(root_julia)) < 1e-10
    retcode = converged ? SciMLBase.ReturnCode.Success : SciMLBase.ReturnCode.Failure
    stats_dict = Dict{Symbol, Any}(:time => t1 - t0)
    if pyhasattr(result, "iterations")
        try stats_dict[:iterations] = pyconvert(Int, result.iterations) catch; end
    end
    if pyhasattr(result, "function_calls")
        try stats_dict[:fevals] = pyconvert(Int, result.function_calls) catch; end
    end
    stats = Optimization.OptimizationStats(; stats_dict...)
    return SciMLBase.build_solution(cache, cache.opt, minimizer, minimum;
                                   original = result,
                                   retcode = retcode,
                                   stats = stats)
end

function SciMLBase.__solve(cache::OptimizationCache{F,RC,LB,UB,LC,UC,S,O,D,P,C}) where 
                          {F,RC,LB,UB,LC,UC,S,O<:ScipyRoot,D,P,C}
    _func = _create_loss(cache, vector_output=true)
    kwargs = Dict{Symbol, Any}()
    kwargs[:method] = cache.opt.method
    if !isnothing(cache.f.grad) && cache.opt.method in ["hybr", "lm"]
        _jac = function(θ)
            θ_julia = ensure_julia_array(θ, eltype(cache.u0))
            fval = cache.f(θ_julia, cache.p)
            if isa(fval, Tuple)
                fval = fval[1]
            end
            if isa(fval, Number)
                fval = [fval]
            end
            m = length(fval)
            n = length(θ_julia)
            jac = zeros(eltype(cache.u0), m, n)
            cache.f.grad(jac, θ_julia, cache.p)
            return jac
        end
        kwargs[:jac] = _jac
    end
    if isa(cache.solver_args, NamedTuple)
        _merge_solver_kwargs!(kwargs, cache.solver_args)
    end
    t0 = time()
    result = nothing
    try
        result = scipy.optimize.root(
            _func,
            cache.u0;
            kwargs...
        )
    catch e
        if e isa PythonCall.Core.PyException
            py_msg = sprint(showerror, e)
            if occursin("Optimization halted by callback", py_msg)
                throw(ErrorException("Optimization halted by callback"))
            elseif occursin("Optimization halted: time limit exceeded", py_msg)
                throw(ErrorException("Optimization halted: time limit exceeded"))
            else
                throw(ErrorException("SciPy root finding failed: $py_msg"))
            end
        else
            rethrow(e)
        end
    end
    if isnothing(result)
        throw(ErrorException("Root finding failed to return a result"))
    end
    t1 = time()
    if pyis(result.x, pybuiltins.None)
        minimizer = fill(NaN, length(cache.u0))
    else
        minimizer = pyconvert(Vector{eltype(cache.u0)}, result.x)
    end
    fun_val = pyconvert(Vector{Float64}, result.fun)
    minimum = sum(abs2, fun_val)
    py_success = pyconvert(Bool, pybool(result.success))
    py_message = safe_get_message(result)
    retcode = py_success ? SciMLBase.ReturnCode.Success : SciMLBase.ReturnCode.Failure
    if retcode != SciMLBase.ReturnCode.Success
        @debug "ScipyRoot convergence: $(py_message)"
    end
    stats = extract_stats(result, t1 - t0)
    return SciMLBase.build_solution(cache, cache.opt, minimizer, minimum;
                                   original = result,
                                   retcode = retcode,
                                   stats = stats)
end

function SciMLBase.__solve(cache::OptimizationCache{F,RC,LB,UB,LC,UC,S,O,D,P,C}) where 
                          {F,RC,LB,UB,LC,UC,S,O<:ScipyLinprog,D,P,C}
    c = cache.f(cache.u0, cache.p)
    if isa(c, Tuple)
        c = c[1]
    end
    if isa(c, Number)
        c = [c]
    end
    bounds = nothing
    if !isnothing(cache.lb) || !isnothing(cache.ub)
        n = length(cache.u0)
        lb = isnothing(cache.lb) ? fill(-Inf, n) : cache.lb
        ub = isnothing(cache.ub) ? fill(Inf, n) : cache.ub
        if length(lb) != n
            lb_new = fill(-Inf, n)
            lb_new[1:min(length(lb), n)] .= lb[1:min(length(lb), n)]
            lb = lb_new
        end
        if length(ub) != n
            ub_new = fill(Inf, n)
            ub_new[1:min(length(ub), n)] .= ub[1:min(length(ub), n)]
            ub = ub_new
        end
        bounds_list = []
        for i in 1:n
            lb_val = isfinite(lb[i]) ? lb[i] : nothing
            ub_val = isfinite(ub[i]) ? ub[i] : nothing
            push!(bounds_list, (lb_val, ub_val))
        end
        bounds = pylist(bounds_list)
    end
    # Allow users to pass constraint matrices via solver kwargs
    A_ub = get(cache.solver_args, :A_ub, nothing)
    b_ub = get(cache.solver_args, :b_ub, nothing)
    A_eq = get(cache.solver_args, :A_eq, nothing)
    b_eq = get(cache.solver_args, :b_eq, nothing)
    if !(isnothing(A_ub) == isnothing(b_ub))
        throw(ArgumentError("Both A_ub and b_ub must be provided together"))
    end
    if !(isnothing(A_eq) == isnothing(b_eq))
        throw(ArgumentError("Both A_eq and b_eq must be provided together"))
    end
    maxiters = Optimization._check_and_convert_maxiters(cache.solver_args.maxiters)
    options = nothing
    if !isnothing(maxiters)
        options = pydict(Dict("maxiter" => maxiters))
    end
    t0 = time()
    result = nothing
    try
        result = scipy.optimize.linprog(
            c,
            A_ub = A_ub,
            b_ub = b_ub,
            A_eq = A_eq,
            b_eq = b_eq,
            bounds = bounds,
            method = cache.opt.method,
            options = options
        )
    catch e
        if e isa PythonCall.Core.PyException
            py_msg = sprint(showerror, e)
            throw(ErrorException("SciPy linear programming failed: $py_msg"))
        else
            rethrow(e)
        end
    end
    if isnothing(result)
        throw(ErrorException("Linear programming failed to return a result"))
    end
    t1 = time()
    if pyis(result.x, pybuiltins.None)
        minimizer = fill(NaN, length(cache.u0))
    else
        minimizer = pyconvert(Vector{eltype(cache.u0)}, result.x)
    end
    minimum = pyis(result.fun, pybuiltins.None) ? NaN : safe_to_float(result.fun)
    py_success = pyconvert(Bool, pybool(result.success))
    py_message = safe_get_message(result)
    if cache.sense === Optimization.MaxSense
        minimum = -minimum
    end
    status = 0
    if pyhasattr(result, "status")
        try
            status = pyconvert(Int, result.status)
        catch
        end
    end
    retcode = scipy_status_to_retcode(status, py_success)
    if retcode != SciMLBase.ReturnCode.Success
        @debug "ScipyLinprog convergence: $(py_message)"
    end
    stats = extract_stats(result, t1 - t0)
    return SciMLBase.build_solution(cache, cache.opt, minimizer, minimum;
                                   original = result,
                                   retcode = retcode,
                                   stats = stats)
end

function SciMLBase.__solve(cache::OptimizationCache{F,RC,LB,UB,LC,UC,S,O,D,P,C}) where 
                          {F,RC,LB,UB,LC,UC,S,O<:ScipyMilp,D,P,C}
    c = cache.f(cache.u0, cache.p)
    if isa(c, Tuple)
        c = c[1]
    end
    if isa(c, Number)
        c = [c]
    end
    n = length(c)
    lb = isnothing(cache.lb) ? fill(-Inf, n) : copy(cache.lb)
    ub = isnothing(cache.ub) ? fill(Inf, n) : copy(cache.ub)
    if length(lb) != n
        lb_new = fill(-Inf, n)
        lb_new[1:min(length(lb), n)] .= lb[1:min(length(lb), n)]
        lb = lb_new
    end
    if length(ub) != n
        ub_new = fill(Inf, n)
        ub_new[1:min(length(ub), n)] .= ub[1:min(length(ub), n)]
        ub = ub_new
    end
    bounds = scipy.optimize.Bounds(lb, ub)
    integrality = get(cache.solver_args, :integrality, nothing)
    A = get(cache.solver_args, :A, nothing)
    lb_con = get(cache.solver_args, :lb_con, nothing)
    ub_con = get(cache.solver_args, :ub_con, nothing)
    constraints = nothing
    if !(isnothing(A) && isnothing(lb_con) && isnothing(ub_con))
        if any(isnothing.((A, lb_con, ub_con)))
            throw(ArgumentError("A, lb_con, and ub_con must all be provided for linear constraints"))
        end
        keep_feasible_flag = get(cache.solver_args, :keep_feasible, false)
        constraints = scipy.optimize.LinearConstraint(A, lb_con, ub_con, keep_feasible = keep_feasible_flag)
    end
    t0 = time()
    result = nothing
    try
        result = scipy.optimize.milp(
            c = c,
            integrality = integrality,
            bounds = bounds,
            constraints = constraints,
            options = nothing
        )
    catch e
        if e isa PythonCall.Core.PyException
            py_msg = sprint(showerror, e)
            throw(ErrorException("SciPy MILP failed: $py_msg"))
        else
            rethrow(e)
        end
    end
    if isnothing(result)
        throw(ErrorException("MILP failed to return a result"))
    end
    t1 = time()
    if pyis(result.x, pybuiltins.None)
        minimizer = fill(NaN, length(cache.u0))
    else
        minimizer = pyconvert(Vector{eltype(cache.u0)}, result.x)
    end
    minimum = pyis(result.fun, pybuiltins.None) ? NaN : safe_to_float(result.fun)
    py_success = pyconvert(Bool, pybool(result.success))
    py_message = safe_get_message(result)
    if cache.sense === Optimization.MaxSense
        minimum = -minimum
    end
    retcode = py_success ? SciMLBase.ReturnCode.Success : SciMLBase.ReturnCode.Failure
    if retcode != SciMLBase.ReturnCode.Success
        @debug "ScipyMilp convergence: $(py_message)"
    end
    stats = extract_stats(result, t1 - t0)
    return SciMLBase.build_solution(cache, cache.opt, minimizer, minimum;
                                   original = result,
                                   retcode = retcode,
                                   stats = stats)
end

function SciMLBase.__solve(cache::OptimizationCache{F,RC,LB,UB,LC,UC,S,O,D,P,C}) where 
                          {F,RC,LB,UB,LC,UC,S,O<:ScipyDifferentialEvolution,D,P,C}
    _loss  = _create_loss(cache)
    bounds = _build_bounds(cache.lb, cache.ub)
    maxiters = Optimization._check_and_convert_maxiters(cache.solver_args.maxiters)
    de_kwargs = Dict{Symbol, Any}()
    de_kwargs[:maxiter]       = isnothing(maxiters) ? 1000 : maxiters
    de_kwargs[:popsize]       = 15
    de_kwargs[:atol]          = 0.0
    de_kwargs[:tol]           = 0.01
    de_kwargs[:mutation]      = (0.5, 1)
    de_kwargs[:recombination] = 0.7
    de_kwargs[:polish]        = true
    de_kwargs[:init]          = "latinhypercube"
    de_kwargs[:updating]      = "immediate"
    de_kwargs[:workers]       = 1
    _merge_solver_kwargs!(de_kwargs, cache.solver_args)
    t0 = time()
    result = nothing
    try
        result = scipy.optimize.differential_evolution(
            _loss,
            bounds;
            de_kwargs...
        )
    catch e
        if e isa PythonCall.Core.PyException
            py_msg = sprint(showerror, e)
            if occursin("Optimization halted by callback", py_msg)
                throw(ErrorException("Optimization halted by callback"))
            elseif occursin("Optimization halted: time limit exceeded", py_msg)
                throw(ErrorException("Optimization halted: time limit exceeded"))
            else
                throw(ErrorException("SciPy optimization failed: $py_msg"))
            end
        else
            rethrow(e)
        end
    end
    if isnothing(result)
        throw(ErrorException("Optimization failed to return a result"))
    end
    t1 = time()
    if pyis(result.x, pybuiltins.None)
        minimizer = fill(NaN, length(cache.u0))
    else
        minimizer = pyconvert(Vector{eltype(cache.u0)}, result.x)
    end
    minimum = safe_to_float(result.fun)
    py_success = pyconvert(Bool, pybool(result.success))
    py_message = safe_get_message(result)
    if cache.sense === Optimization.MaxSense
        minimum = -minimum
    end
    retcode = py_success ? SciMLBase.ReturnCode.Success : SciMLBase.ReturnCode.Failure
    if retcode != SciMLBase.ReturnCode.Success
        @debug "ScipyDifferentialEvolution convergence: $(py_message)"
    end
    stats = extract_stats(result, t1 - t0)
    return SciMLBase.build_solution(cache, cache.opt, minimizer, minimum;
                                   original = result,
                                   retcode = retcode,
                                   stats = stats)
end

function SciMLBase.__solve(cache::OptimizationCache{F,RC,LB,UB,LC,UC,S,O,D,P,C}) where 
                          {F,RC,LB,UB,LC,UC,S,O<:ScipyBasinhopping,D,P,C}
    _loss = _create_loss(cache)
    maxiters = Optimization._check_and_convert_maxiters(cache.solver_args.maxiters)
    bh_kwargs = Dict{Symbol, Any}()
    bh_kwargs[:niter]     = isnothing(maxiters) ? 100 : maxiters
    bh_kwargs[:T]         = 1.0
    bh_kwargs[:stepsize]  = 0.5
    bh_kwargs[:interval]  = 50
    _merge_solver_kwargs!(bh_kwargs, cache.solver_args)
    t0 = time()
    result = nothing
    try
        result = scipy.optimize.basinhopping(
            _loss,
            cache.u0;
            bh_kwargs...
        )
    catch e
        if e isa PythonCall.Core.PyException
            py_msg = sprint(showerror, e)
            if occursin("Optimization halted by callback", py_msg)
                throw(ErrorException("Optimization halted by callback"))
            elseif occursin("Optimization halted: time limit exceeded", py_msg)
                throw(ErrorException("Optimization halted: time limit exceeded"))
            else
                throw(ErrorException("SciPy optimization failed: $py_msg"))
            end
        else
            rethrow(e)
        end
    end
    if isnothing(result)
        throw(ErrorException("Optimization failed to return a result"))
    end
    t1 = time()
    if pyis(result.x, pybuiltins.None)
        minimizer = fill(NaN, length(cache.u0))
    else
        minimizer = pyconvert(Vector{eltype(cache.u0)}, result.x)
    end
    minimum = safe_to_float(result.fun)
    lowest_result = result.lowest_optimization_result
    py_success = pyconvert(Bool, pybool(lowest_result.success))
    py_message = safe_get_message(lowest_result)
    if cache.sense === Optimization.MaxSense
        minimum = -minimum
    end
    retcode = py_success ? SciMLBase.ReturnCode.Success : SciMLBase.ReturnCode.Failure
    if retcode != SciMLBase.ReturnCode.Success
        @debug "ScipyBasinhopping convergence: $(py_message)"
    end
    stats = extract_stats(lowest_result, t1 - t0)
    return SciMLBase.build_solution(cache, cache.opt, minimizer, minimum;
                                   original = result,
                                   retcode = retcode,
                                   stats = stats)
end

function SciMLBase.__solve(cache::OptimizationCache{F,RC,LB,UB,LC,UC,S,O,D,P,C}) where 
                          {F,RC,LB,UB,LC,UC,S,O<:ScipyDualAnnealing,D,P,C}
    _loss  = _create_loss(cache)
    bounds = _build_bounds(cache.lb, cache.ub)
    da_kwargs = Dict{Symbol, Any}()
    da_kwargs[:maxiter]         = begin
        mi = Optimization._check_and_convert_maxiters(cache.solver_args.maxiters)
        isnothing(mi) ? 1000 : mi
    end
    da_kwargs[:initial_temp]    = 5230.0
    da_kwargs[:restart_temp_ratio] = 2e-5
    da_kwargs[:visit]           = 2.62
    da_kwargs[:accept]          = -5.0
    da_kwargs[:maxfun]          = 1e7
    da_kwargs[:no_local_search] = false
    _merge_solver_kwargs!(da_kwargs, cache.solver_args)
    t0 = time()
    result = nothing
    try
        result = scipy.optimize.dual_annealing(
            _loss,
            bounds;
            da_kwargs...
        )
    catch e
        if e isa PythonCall.Core.PyException
            py_msg = sprint(showerror, e)
            if occursin("Optimization halted by callback", py_msg)
                throw(ErrorException("Optimization halted by callback"))
            elseif occursin("Optimization halted: time limit exceeded", py_msg)
                throw(ErrorException("Optimization halted: time limit exceeded"))
            else
                throw(ErrorException("SciPy optimization failed: $py_msg"))
            end
        else
            rethrow(e)
        end
    end
    if isnothing(result)
        throw(ErrorException("Optimization failed to return a result"))
    end
    t1 = time()
    if pyis(result.x, pybuiltins.None)
        minimizer = fill(NaN, length(cache.u0))
    else
        minimizer = pyconvert(Vector{eltype(cache.u0)}, result.x)
    end
    minimum = safe_to_float(result.fun)
    py_success = pyconvert(Bool, pybool(result.success))
    py_message = safe_get_message(result)
    if cache.sense === Optimization.MaxSense
        minimum = -minimum
    end
    retcode = py_success ? SciMLBase.ReturnCode.Success : SciMLBase.ReturnCode.Failure
    if retcode != SciMLBase.ReturnCode.Success
        @debug "ScipyDualAnnealing convergence: $(py_message)"
    end
    stats = extract_stats(result, t1 - t0)
    return SciMLBase.build_solution(cache, cache.opt, minimizer, minimum;
                                   original = result,
                                   retcode = retcode,
                                   stats = stats)
end

function SciMLBase.__solve(cache::OptimizationCache{F,RC,LB,UB,LC,UC,S,O,D,P,C}) where 
                          {F,RC,LB,UB,LC,UC,S,O<:ScipyShgo,D,P,C}
    local cons_cache = nothing
    if !isnothing(cache.f.cons) && !isnothing(cache.lcons)
        cons_cache = zeros(eltype(cache.u0), length(cache.lcons))
    end
    _loss = _create_loss(cache)
    bounds = _build_bounds(cache.lb, cache.ub)
    constraints = nothing
    if !isnothing(cons_cache)
        cons_list = []
        _cons_func = function(θ)
            θ_julia = ensure_julia_array(θ, eltype(cache.u0))
            cons_cache .= zero(eltype(cons_cache))
            if hasmethod(cache.f.cons, Tuple{typeof(cons_cache), typeof(θ_julia), typeof(cache.p)})
                cache.f.cons(cons_cache, θ_julia, cache.p)
            else
                cache.f.cons(cons_cache, θ_julia)
            end
            return cons_cache
        end
        for i in 1:length(cache.lcons)
            if isfinite(cache.lcons[i])
                cons_func_i = let i=i, _cons_func=_cons_func
                    θ -> _cons_func(θ)[i] - cache.lcons[i]
                end
                push!(cons_list, pydict(Dict("type" => "ineq", "fun" => cons_func_i)))
            end
        end
        for i in 1:length(cache.ucons)
            if isfinite(cache.ucons[i])
                cons_func_i = let i=i, _cons_func=_cons_func
                    θ -> cache.ucons[i] - _cons_func(θ)[i]
                end
                push!(cons_list, pydict(Dict("type" => "ineq", "fun" => cons_func_i)))
            end
        end
        constraints = pylist(cons_list)
    end
    shgo_kwargs = Dict{Symbol, Any}()
    shgo_kwargs[:n] = 100
    shgo_kwargs[:iters] = 1
    shgo_kwargs[:sampling_method] = "simplicial"
    _merge_solver_kwargs!(shgo_kwargs, cache.solver_args)
    t0 = time()
    result = nothing
    try
        result = scipy.optimize.shgo(
            _loss,
            bounds;
            args = (),
            constraints = constraints,
            shgo_kwargs...
        )
    catch e
        if e isa PythonCall.Core.PyException
            py_msg = sprint(showerror, e)
            if occursin("Optimization halted by callback", py_msg)
                throw(ErrorException("Optimization halted by callback"))
            elseif occursin("Optimization halted: time limit exceeded", py_msg)
                throw(ErrorException("Optimization halted: time limit exceeded"))
            else
                throw(ErrorException("SciPy optimization failed: $py_msg"))
            end
        else
            rethrow(e)
        end
    end
    if isnothing(result)
        throw(ErrorException("Optimization failed to return a result"))
    end
    t1 = time()
    if pyis(result.x, pybuiltins.None)
        minimizer = fill(NaN, length(cache.u0))
    else
        minimizer = pyconvert(Vector{eltype(cache.u0)}, result.x)
    end
    minimum = safe_to_float(result.fun)
    py_success = pyconvert(Bool, pybool(result.success))
    py_message = safe_get_message(result)
    if cache.sense === Optimization.MaxSense
        minimum = -minimum
    end
    retcode = py_success ? SciMLBase.ReturnCode.Success : SciMLBase.ReturnCode.Failure
    if retcode != SciMLBase.ReturnCode.Success
        @debug "ScipyShgo convergence: $(py_message)"
    end
    stats = extract_stats(result, t1 - t0)
    return SciMLBase.build_solution(cache, cache.opt, minimizer, minimum;
                                   original = result,
                                   retcode = retcode,
                                   stats = stats)
end

function SciMLBase.__solve(cache::OptimizationCache{F,RC,LB,UB,LC,UC,S,O,D,P,C}) where 
                          {F,RC,LB,UB,LC,UC,S,O<:ScipyDirect,D,P,C}
    _loss  = _create_loss(cache)
    bounds = _build_bounds(cache.lb, cache.ub)
    maxiters = Optimization._check_and_convert_maxiters(cache.solver_args.maxiters)
    direct_kwargs = Dict{Symbol, Any}()
    direct_kwargs[:eps]            = 0.0001
    direct_kwargs[:maxiter]        = isnothing(maxiters) ? 1000 : maxiters
    direct_kwargs[:locally_biased] = true
    direct_kwargs[:vol_tol]        = 1e-16
    direct_kwargs[:len_tol]        = 1e-6
    _merge_solver_kwargs!(direct_kwargs, cache.solver_args)
    t0 = time()
    result = nothing
    try
        result = scipy.optimize.direct(
            _loss,
            bounds;
            direct_kwargs...
        )
    catch e
        if e isa PythonCall.Core.PyException
            py_msg = sprint(showerror, e)
            if occursin("Optimization halted by callback", py_msg)
                throw(ErrorException("Optimization halted by callback"))
            elseif occursin("Optimization halted: time limit exceeded", py_msg)
                throw(ErrorException("Optimization halted: time limit exceeded"))
            else
                throw(ErrorException("SciPy optimization failed: $py_msg"))
            end
        else
            rethrow(e)
        end
    end
    if isnothing(result)
        throw(ErrorException("Optimization failed to return a result"))
    end
    t1 = time()
    if pyis(result.x, pybuiltins.None)
        minimizer = fill(NaN, length(cache.u0))
    else
        minimizer = pyconvert(Vector{eltype(cache.u0)}, result.x)
    end
    minimum = safe_to_float(result.fun)
    py_success = pyconvert(Bool, pybool(result.success))
    py_message = safe_get_message(result)
    if cache.sense === Optimization.MaxSense
        minimum = -minimum
    end
    retcode = py_success ? SciMLBase.ReturnCode.Success : SciMLBase.ReturnCode.Failure
    if retcode != SciMLBase.ReturnCode.Success
        @debug "ScipyDirect convergence: $(py_message)"
    end
    stats = extract_stats(result, t1 - t0)
    return SciMLBase.build_solution(cache, cache.opt, minimizer, minimum;
                                   original = result,
                                   retcode = retcode,
                                   stats = stats)
end

function SciMLBase.__solve(cache::OptimizationCache{F,RC,LB,UB,LC,UC,S,O,D,P,C}) where 
                      {F,RC,LB,UB,LC,UC,S,O<:ScipyBrute,D,P,C}
    _loss = _create_loss(cache)
    ranges = _build_bounds(cache.lb, cache.ub)
    brute_kwargs = Dict{Symbol, Any}()
    brute_kwargs[:Ns] = 20
    brute_kwargs[:full_output] = true
    brute_kwargs[:finish] = scipy.optimize.fmin
    brute_kwargs[:workers] = 1
    _merge_solver_kwargs!(brute_kwargs, cache.solver_args)
    t0 = time()
    result = nothing
    try
        result = scipy.optimize.brute(
            _loss,
            ranges;
            brute_kwargs...
        )
    catch e
        if e isa PythonCall.Core.PyException
            py_msg = sprint(showerror, e)
            if occursin("Optimization halted by callback", py_msg)
                throw(ErrorException("Optimization halted by callback"))
            elseif occursin("Optimization halted: time limit exceeded", py_msg)
                throw(ErrorException("Optimization halted: time limit exceeded"))
            else
                throw(ErrorException("SciPy optimization failed: $py_msg"))
            end
        else
            rethrow(e)
        end
    end
    if isnothing(result)
        throw(ErrorException("Optimization failed to return a result"))
    end
    t1 = time()
    if pyis(result[0], pybuiltins.None)
        minimizer = fill(NaN, length(cache.u0))
    else
        minimizer = pyconvert(Vector{eltype(cache.u0)}, result[0])
    end
    minimum = safe_to_float(result[1])
    if cache.sense === Optimization.MaxSense
        minimum = -minimum
    end
    retcode = SciMLBase.ReturnCode.Success
    stats = Optimization.OptimizationStats(; time = t1 - t0)
    return SciMLBase.build_solution(cache, cache.opt, minimizer, minimum;
                                   original = result,
                                   retcode = retcode,
                                   stats = stats)
end

export ScipyMinimize, ScipyNelderMead, ScipyPowell, ScipyCG, ScipyBFGS, ScipyNewtonCG,
       ScipyLBFGSB, ScipyTNC, ScipyCOBYLA, ScipyCOBYQA, ScipySLSQP, ScipyTrustConstr,
       ScipyDogleg, ScipyTrustNCG, ScipyTrustKrylov, ScipyTrustExact,
       ScipyMinimizeScalar, ScipyBrent, ScipyBounded, ScipyGolden,
       ScipyLeastSquares, ScipyLeastSquaresTRF, ScipyLeastSquaresDogbox, ScipyLeastSquaresLM,
       ScipyRootScalar, ScipyRoot, ScipyLinprog, ScipyMilp,
       ScipyDifferentialEvolution, ScipyBasinhopping, ScipyDualAnnealing, 
       ScipyShgo, ScipyDirect, ScipyBrute

# Wrap the user's Julia objective so it matches what SciPy expects.
function _create_loss(cache; vector_output::Bool = false)
    maxtime = get(cache.solver_args, :maxtime, nothing)
    start_time = !isnothing(maxtime) ? time() : 0.0
    if vector_output
        return function (θ)
            if !isnothing(maxtime) && (time() - start_time) > maxtime
                error("Optimization halted: time limit exceeded")
            end
            θ_julia = ensure_julia_array(θ, eltype(cache.u0))
            x = cache.f(θ_julia, cache.p)
            if isa(x, Tuple)
                x = x
            elseif isa(x, Number)
                x = (x,)
            end
            opt_state = Optimization.OptimizationState(u = θ_julia, objective = sum(abs2, x))
            if cache.callback(opt_state, x...)
                error("Optimization halted by callback")
            end
          
            arr = cache.sense === Optimization.MaxSense ? -x : x
            return arr
        end
    else
        return function (θ)
            if !isnothing(maxtime) && (time() - start_time) > maxtime
                error("Optimization halted: time limit exceeded")
            end
            θ_julia = ensure_julia_array(θ, eltype(cache.u0))
            x = cache.f(θ_julia, cache.p)
            if isa(x, Tuple)
                x = x
            elseif isa(x, Number)
                x = (x,)
            end
            opt_state = Optimization.OptimizationState(u = θ_julia, objective = x[1])
            if cache.callback(opt_state, x...)
                error("Optimization halted by callback")
            end
            return cache.sense === Optimization.MaxSense ? -x[1] : x[1]
        end
    end
end

# These solver-args are handled specially elsewhere, so we skip them here.
const _DEFAULT_EXCLUDE = (
    :maxiters, :maxtime, :abstol, :reltol, :callback, :progress, :cons_tol,
    :jac_sparsity, :keep_feasible, :hess_update
)

# Moving the remaining kwargs into a Dict that we pass straight to SciPy.
function _merge_solver_kwargs!(dest::AbstractDict, solver_args; exclude = _DEFAULT_EXCLUDE)
    if isa(solver_args, NamedTuple)
        for (k, v) in pairs(solver_args)
            k in exclude && continue
            isnothing(v) && continue
            dest[convert(keytype(dest), k)] = v
        end
    end
    return dest
end

function _build_bounds(lb::AbstractVector, ub::AbstractVector)
    return pylist([pytuple([lb[i], ub[i]]) for i in eachindex(lb)])
end

struct ResidualObjective{R}
    residual::R
end

(r::ResidualObjective)(u, p) = sum(abs2, r.residual(u, p))

end

