module OptimizationSciPy

using Reexport
@reexport using Optimization
using Optimization.SciMLBase
using PythonCall

const scipy = PythonCall.pynew()

function __init__()
    PythonCall.pycopy!(scipy, pyimport("scipy"))
end

function ensure_julia_array(x)
    x isa Vector{Float64} && return x
    return convert(Vector{Float64}, x isa Py ? pyconvert(Vector, x) : x)
end

function safe_get_message(result)
    try
        pyconvert(String, result.message)
    catch
        "Optimization completed"
    end
end

function safe_to_float(x)
    isa(x, Float64) && return x
    
    try
        return pyconvert(Float64, x)
    catch
        try
            item_val = x.item()
            return pyconvert(Float64, item_val)
        catch
            try
                return pyconvert(Float64, pybuiltins.float(x))
            catch
                error("Cannot convert Python object to Float64: $(x)")
            end
        end
    end
end

function extract_stats(result, time_elapsed)
    stats_dict = Dict{Symbol, Any}(
        :iterations => 0,
        :time => time_elapsed,
        :fevals => 0,
        :gevals => 0,
        :hevals => 0
    )
    
    if pyhasattr(result, "nit")
        try
            stats_dict[:iterations] = pyconvert(Int, result.nit)
        catch
        end
    end
    
    if pyhasattr(result, "nfev")
        try
            stats_dict[:fevals] = pyconvert(Int, result.nfev)
        catch
        end
    end
    
    if pyhasattr(result, "njev")
        try
            stats_dict[:gevals] = pyconvert(Int, result.njev)
        catch
        end
    elseif pyhasattr(result, "ngrad")
        try
            stats_dict[:gevals] = pyconvert(Int, result.ngrad)
        catch
        end
    end
    
    if pyhasattr(result, "nhev")
        try
            stats_dict[:hevals] = pyconvert(Int, result.nhev)
        catch
        end
    end
    
    return Optimization.OptimizationStats(; stats_dict...)
end

"""
    ScipyMinimize(method::String)

Wrapper for scipy.optimize.minimize with specified method.

Available methods:
- "Nelder-Mead": Gradient-free simplex method
- "Powell": Gradient-free Powell's method
- "CG": Conjugate gradient
- "BFGS": Quasi-Newton BFGS
- "Newton-CG": Newton conjugate gradient (requires Hessian)
- "L-BFGS-B": Limited memory BFGS with bounds
- "TNC": Truncated Newton with bounds
- "COBYLA": Constrained optimization by linear approximation (gradient-free)
- "SLSQP": Sequential least squares programming (supports constraints)
- "trust-constr": Trust-region constrained optimization
- "dogleg": Dogleg trust-region (requires Hessian)
- "trust-ncg": Trust-region Newton conjugate gradient (requires Hessian)
- "trust-krylov": Trust-region Krylov subspace (requires Hessian)
- "trust-exact": Trust-region with exact solution (requires Hessian)
"""
struct ScipyMinimize
    method::String
    
    function ScipyMinimize(method::String)
        valid_methods = ["Nelder-Mead", "Powell", "CG", "BFGS", "Newton-CG",
                        "L-BFGS-B", "TNC", "COBYLA", "SLSQP",
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
ScipySLSQP() = ScipyMinimize("SLSQP")
ScipyTrustConstr() = ScipyMinimize("trust-constr")
ScipyDogleg() = ScipyMinimize("dogleg")
ScipyTrustNCG() = ScipyMinimize("trust-ncg")
ScipyTrustKrylov() = ScipyMinimize("trust-krylov")
ScipyTrustExact() = ScipyMinimize("trust-exact")

"""
    ScipyDifferentialEvolution()

Global optimization using differential evolution algorithm.

Differential evolution is a stochastic population based method that is useful 
for global optimization problems. At each pass through the population the algorithm 
mutates each candidate solution by mixing with other candidate solutions to create 
a trial candidate.

Requires bounds on all variables.
"""
struct ScipyDifferentialEvolution end

"""
    ScipyBasinhopping()

Global optimization using basin-hopping algorithm.

Basin-hopping is a stochastic algorithm which attempts to find the global minimum
of a function by combining random jumping with local minimization.
"""
struct ScipyBasinhopping end

"""
    ScipyDualAnnealing()

Global optimization using dual annealing.

Dual annealing combines simulated annealing with a local search at each iteration.
It uses a visiting distribution to sample new points and an acceptance distribution
to decide whether to accept new points.

Requires bounds on all variables.
"""
struct ScipyDualAnnealing end

"""
    ScipyShgo()

Global optimization using simplicial homology global optimization (SHGO).

SHGO is a global optimization algorithm that uses simplicial homology theory to
find all local minima and the global minimum. Supports constraints.

Requires bounds on all variables.
"""
struct ScipyShgo end

"""
    ScipyDirect()

Global optimization using DIRECT algorithm.

DIRECT is a deterministic global optimization algorithm that systematically
divides the search space into smaller hyperrectangles.

Requires bounds on all variables.
"""
struct ScipyDirect end

"""
    ScipyBrute()

Global optimization using brute force grid search.

Brute force optimization by evaluating the objective function on a grid
of points. Best used for low-dimensional problems.

Requires bounds on all variables.
"""
struct ScipyBrute end

for opt_type in [:ScipyMinimize, :ScipyDifferentialEvolution, :ScipyBasinhopping, 
                  :ScipyDualAnnealing, :ScipyShgo, :ScipyDirect, :ScipyBrute]
    @eval begin
        SciMLBase.allowsbounds(::$opt_type) = true
        SciMLBase.supports_opt_cache_interface(::$opt_type) = true
    end
end

function SciMLBase.requiresgradient(opt::ScipyMinimize)
    gradient_free = ["Nelder-Mead", "Powell", "COBYLA"]
    return !(opt.method in gradient_free)
end

for opt_type in [:ScipyDifferentialEvolution, :ScipyBasinhopping, 
                  :ScipyDualAnnealing, :ScipyShgo, :ScipyDirect, :ScipyBrute]
    @eval SciMLBase.requiresgradient(::$opt_type) = false
end

function SciMLBase.requireshessian(opt::ScipyMinimize)
    hessian_methods = ["Newton-CG", "dogleg", "trust-ncg", "trust-exact", "trust-krylov"]
    return opt.method in hessian_methods
end

function SciMLBase.allowsconstraints(opt::ScipyMinimize)
    return opt.method in ["SLSQP", "trust-constr", "COBYLA"]
end

function SciMLBase.requiresconsjac(opt::ScipyMinimize)
    return opt.method in ["SLSQP", "trust-constr"]
end

SciMLBase.allowsconstraints(::ScipyShgo) = true

function SciMLBase.__init(prob::SciMLBase.OptimizationProblem, opt::Union{ScipyMinimize, 
                         ScipyDifferentialEvolution, ScipyBasinhopping, ScipyDualAnnealing, 
                         ScipyShgo, ScipyDirect, ScipyBrute};
                         cons_tol = 1e-6,
                         callback = (args...) -> (false),
                         progress = false, 
                         kwargs...)
    
    requires_bounds = opt isa Union{ScipyDifferentialEvolution, ScipyDirect, ScipyDualAnnealing, ScipyBrute}
    if requires_bounds && (isnothing(prob.lb) || isnothing(prob.ub))
        throw(SciMLBase.IncompatibleOptimizerError("$(typeof(opt)) requires bounds"))
    end
    
    if !isnothing(prob.lb) && !isnothing(prob.ub)
        @assert length(prob.lb) == length(prob.ub) "Bounds must have the same length"
        @assert all(prob.lb .<= prob.ub) "Lower bounds must be less than or equal to upper bounds"
    end
    
    return OptimizationCache(prob, opt; cons_tol, callback, progress, kwargs...)
end

function SciMLBase.__solve(cache::OptimizationCache{F,RC,LB,UB,LC,UC,S,O,D,P,C}) where 
                          {F,RC,LB,UB,LC,UC,S,O<:ScipyMinimize,D,P,C}
    
    _loss = function(θ)
        θ_julia = ensure_julia_array(θ)
        x = cache.f(θ_julia, cache.p)
        opt_state = Optimization.OptimizationState(u = θ_julia, objective = x[1])
        
        if cache.callback(opt_state, x...)
            error("Optimization halted by callback")
        end
        
        return cache.sense === Optimization.MaxSense ? -x[1] : x[1]
    end
    
    maxiters = Optimization._check_and_convert_maxiters(cache.solver_args.maxiters)
    abstol = cache.solver_args.abstol
    reltol = cache.solver_args.reltol
    
    options = Dict{String, Any}()
    if !isnothing(maxiters)
        if cache.opt.method in ["trust-constr", "dogleg", "trust-ncg", "trust-krylov", "trust-exact"]
            options["maxiter"] = max(maxiters, 5000)
        else
            options["maxiter"] = maxiters
        end
    end
    if !isnothing(abstol)
        if cache.opt.method in ["Nelder-Mead", "Powell", "L-BFGS-B", "TNC", "SLSQP", "trust-constr"]
            options["ftol"] = abstol
        end
    end
    if !isnothing(reltol)
        if cache.opt.method in ["CG", "BFGS", "Newton-CG", "L-BFGS-B", "TNC", "SLSQP", "trust-constr"]
            options["gtol"] = reltol
        end
    end
    
    if cache.opt.method == "trust-constr"
        options["initial_tr_radius"] = 1.0
        options["verbose"] = 0
        options["finite_diff_rel_step"] = 1e-8
    end
    
    if isa(cache.solver_args, NamedTuple)
        for (fname, fval) in pairs(cache.solver_args)
            if fname ∉ (:maxiters, :maxtime, :abstol, :reltol, :callback, :progress, :cons_tol)
                if !isnothing(fval)
                    if cache.opt.method == "trust-constr" || fname ∉ (:verbose, :max_tr_radius, :initial_tr_radius)
                        options[string(fname)] = fval
                    end
                end
            end
        end
    end
    
    jac = nothing
    if SciMLBase.requiresgradient(cache.opt) && !isnothing(cache.f.grad)
        _grad = function(θ)
            θ_julia = ensure_julia_array(θ)
            grad = zeros(Float64, length(θ_julia))
            cache.f.grad(grad, θ_julia, cache.p)
            return cache.sense === Optimization.MaxSense ? -grad : grad
        end
        jac = _grad
    end
    
    hess = nothing
    if SciMLBase.requireshessian(cache.opt)
        if !isnothing(cache.f.hess)
            _hess = function(θ)
                θ_julia = ensure_julia_array(θ)
                H = zeros(Float64, length(θ_julia), length(θ_julia))
                cache.f.hess(H, θ_julia, cache.p)
                return cache.sense === Optimization.MaxSense ? -H : H
            end
            hess = _hess
        else
            if cache.opt.method in ["trust-constr", "dogleg", "trust-ncg", "trust-krylov", "trust-exact"]
                options["hess"] = "BFGS"
                if !isnothing(maxiters)
                    options["maxiter"] = max(maxiters, 5000)
                end
            else
                throw(ArgumentError("Method $(cache.opt.method) requires Hessian but none was provided"))
            end
        end
    end
    
    bounds = nothing
    if !isnothing(cache.lb) && !isnothing(cache.ub)
        if cache.opt.method in ["L-BFGS-B", "TNC", "SLSQP", "trust-constr", "COBYLA"]
            bounds = scipy.optimize.Bounds(cache.lb, cache.ub)
        end
    end
    
    constraints = pylist([])
    if SciMLBase.allowsconstraints(cache.opt)
        if !isnothing(cache.f.cons)
            lcons = isnothing(cache.lcons) ? fill(-Inf, 0) : cache.lcons
            ucons = isnothing(cache.ucons) ? fill(Inf, 0) : cache.ucons
            
            if length(lcons) > 0
                cons_cache = zeros(Float64, length(lcons))
                
                _cons_func = function(θ)
                    θ_julia = ensure_julia_array(θ)
                    if hasmethod(cache.f.cons, Tuple{typeof(cons_cache), typeof(θ_julia), typeof(cache.p)})
                        cache.f.cons(cons_cache, θ_julia, cache.p)
                    else
                        cache.f.cons(cons_cache, θ_julia)
                    end
                    return cons_cache
                end
                
                cons_jac = "2-point"
                if SciMLBase.requiresconsjac(cache.opt) && !isnothing(cache.f.cons_j)
                    cons_j_cache = zeros(Float64, length(lcons), length(cache.u0))
                    _cons_jac = function(θ)
                        θ_julia = ensure_julia_array(θ)
                        if hasmethod(cache.f.cons_j, Tuple{typeof(cons_j_cache), typeof(θ_julia), typeof(cache.p)})
                            cache.f.cons_j(cons_j_cache, θ_julia, cache.p)
                        else
                            cache.f.cons_j(cons_j_cache, θ_julia)
                        end
                        return cons_j_cache
                    end
                    if cache.opt.method in ["SLSQP", "trust-constr"]
                        cons_jac = _cons_jac
                    end
                end
                
                if cache.opt.method == "trust-constr"
                    options["finite_diff_rel_step"] = 1e-8
                    options["xtol"] = 1e-8
                    options["gtol"] = 1e-8
                elseif cache.opt.method == "SLSQP"
                    options["ftol"] = 1e-8
                    options["eps"] = 1e-8
                end
                
                nlc = scipy.optimize.NonlinearConstraint(
                    _cons_func, 
                    lcons, 
                    ucons,
                    jac = cons_jac
                )
                constraints = pylist([nlc])
            end
        end
    elseif !isnothing(cache.f.cons)
        throw(ArgumentError("Method $(cache.opt.method) does not support constraints. Use SLSQP, trust-constr, or COBYLA instead."))
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
            err_str = string(e.val)
            if occursin("Optimization halted by callback", err_str)
                throw(ErrorException("Optimization halted by callback"))
            else
                rethrow(e)
            end
        elseif e isa ErrorException && occursin("Optimization halted by callback", e.msg)
            throw(e)
        else
            rethrow(e)
        end
    end
    
    if isnothing(result)
        throw(ErrorException("Optimization failed to return a result"))
    end
    
    t1 = time()
    
    minimizer = pyconvert(Vector{Float64}, result.x)
    minimum = safe_to_float(result.fun)
    py_success = pyconvert(Bool, pybool(result.success))
    py_message = safe_get_message(result)
    
    if cache.sense === Optimization.MaxSense
        minimum = -minimum
    end
    
    retcode = py_success ? SciMLBase.ReturnCode.Success : SciMLBase.ReturnCode.Failure
    
    if retcode == SciMLBase.ReturnCode.Failure
        @warn "ScipyMinimize failed to converge: $(py_message)"
    end
    
    stats = extract_stats(result, t1 - t0)
    
    return SciMLBase.build_solution(cache, cache.opt, minimizer, minimum;
                                   original = result,
                                   retcode = retcode,
                                   stats = stats)
end

function SciMLBase.__solve(cache::OptimizationCache{F,RC,LB,UB,LC,UC,S,O,D,P,C}) where 
                          {F,RC,LB,UB,LC,UC,S,O<:ScipyDifferentialEvolution,D,P,C}
    
    _loss = function(θ)
        θ_julia = ensure_julia_array(θ)
        x = cache.f(θ_julia, cache.p)
        opt_state = Optimization.OptimizationState(u = θ_julia, objective = x[1])
        if cache.callback(opt_state, x...)
            error("Optimization halted by callback")
        end
        return cache.sense === Optimization.MaxSense ? -x[1] : x[1]
    end
    
    bounds = pylist([pytuple([cache.lb[i], cache.ub[i]]) for i in 1:length(cache.lb)])
    
    maxiters = Optimization._check_and_convert_maxiters(cache.solver_args.maxiters)
    
    de_kwargs = Dict{Symbol, Any}()
    de_kwargs[:maxiter] = isnothing(maxiters) ? 1000 : maxiters
    de_kwargs[:popsize] = 15
    de_kwargs[:atol] = 0.0
    de_kwargs[:tol] = 0.01
    de_kwargs[:mutation] = (0.5, 1)
    de_kwargs[:recombination] = 0.7
    de_kwargs[:polish] = true
    de_kwargs[:init] = "latinhypercube"
    de_kwargs[:updating] = "immediate"
    de_kwargs[:workers] = 1
    
    if isa(cache.solver_args, NamedTuple)
        for (fname, fval) in pairs(cache.solver_args)
            if fname ∉ (:maxiters, :maxtime, :abstol, :reltol, :callback, :progress, :cons_tol)
                if !isnothing(fval)
                    de_kwargs[fname] = fval
                end
            end
        end
    end
    
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
            err_str = string(e.val)
            if occursin("Optimization halted by callback", err_str)
                throw(ErrorException("Optimization halted by callback"))
            else
                rethrow(e)
            end
        else
            rethrow(e)
        end
    end
    
    if isnothing(result)
        throw(ErrorException("Optimization failed to return a result"))
    end
    
    t1 = time()
    
    minimizer = pyconvert(Vector{Float64}, result.x)
    minimum = safe_to_float(result.fun)
    py_success = pyconvert(Bool, pybool(result.success))
    py_message = safe_get_message(result)
    
    if cache.sense === Optimization.MaxSense
        minimum = -minimum
    end
    
    retcode = py_success ? SciMLBase.ReturnCode.Success : SciMLBase.ReturnCode.Failure
    
    if retcode == SciMLBase.ReturnCode.Failure
        @warn "ScipyDifferentialEvolution failed to converge: $(py_message)"
    end
    
    stats = extract_stats(result, t1 - t0)
    
    return SciMLBase.build_solution(cache, cache.opt, minimizer, minimum;
                                   original = result,
                                   retcode = retcode,
                                   stats = stats)
end

function SciMLBase.__solve(cache::OptimizationCache{F,RC,LB,UB,LC,UC,S,O,D,P,C}) where 
                          {F,RC,LB,UB,LC,UC,S,O<:ScipyBasinhopping,D,P,C}
    
    _loss = function(θ)
        θ_julia = ensure_julia_array(θ)
        x = cache.f(θ_julia, cache.p)
        opt_state = Optimization.OptimizationState(u = θ_julia, objective = x[1])
        if cache.callback(opt_state, x...)
            error("Optimization halted by callback")
        end
        return cache.sense === Optimization.MaxSense ? -x[1] : x[1]
    end
    
    maxiters = Optimization._check_and_convert_maxiters(cache.solver_args.maxiters)
    
    bh_kwargs = Dict{Symbol, Any}()
    bh_kwargs[:niter] = isnothing(maxiters) ? 100 : maxiters
    bh_kwargs[:T] = 1.0
    bh_kwargs[:stepsize] = 0.5
    bh_kwargs[:interval] = 50
    
    if isa(cache.solver_args, NamedTuple)
        for (fname, fval) in pairs(cache.solver_args)
            if fname ∉ (:maxiters, :maxtime, :abstol, :reltol, :callback, :progress, :cons_tol)
                if !isnothing(fval)
                    bh_kwargs[fname] = fval
                end
            end
        end
    end
    
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
            err_str = string(e.val)
            if occursin("Optimization halted by callback", err_str)
                throw(ErrorException("Optimization halted by callback"))
            else
                rethrow(e)
            end
        else
            rethrow(e)
        end
    end
    
    if isnothing(result)
        throw(ErrorException("Optimization failed to return a result"))
    end
    
    t1 = time()
    
    minimizer = pyconvert(Vector{Float64}, result.x)
    minimum = safe_to_float(result.fun)
    lowest_result = result.lowest_optimization_result
    py_success = pyconvert(Bool, pybool(lowest_result.success))
    py_message = safe_get_message(lowest_result)
    
    if cache.sense === Optimization.MaxSense
        minimum = -minimum
    end
    
    retcode = py_success ? SciMLBase.ReturnCode.Success : SciMLBase.ReturnCode.Failure
    
    if retcode == SciMLBase.ReturnCode.Failure
        @warn "ScipyBasinhopping failed to converge: $(py_message)"
    end
    
    stats = extract_stats(lowest_result, t1 - t0)
    
    return SciMLBase.build_solution(cache, cache.opt, minimizer, minimum;
                                   original = result,
                                   retcode = retcode,
                                   stats = stats)
end

function SciMLBase.__solve(cache::OptimizationCache{F,RC,LB,UB,LC,UC,S,O,D,P,C}) where 
                          {F,RC,LB,UB,LC,UC,S,O<:ScipyDualAnnealing,D,P,C}
    
    _loss = function(θ)
        θ_julia = ensure_julia_array(θ)
        x = cache.f(θ_julia, cache.p)
        opt_state = Optimization.OptimizationState(u = θ_julia, objective = x[1])
        if cache.callback(opt_state, x...)
            error("Optimization halted by callback")
        end
        return cache.sense === Optimization.MaxSense ? -x[1] : x[1]
    end
    
    bounds = pylist([pytuple([cache.lb[i], cache.ub[i]]) for i in 1:length(cache.lb)])
    
    maxiters = Optimization._check_and_convert_maxiters(cache.solver_args.maxiters)
    
    da_kwargs = Dict{Symbol, Any}()
    da_kwargs[:maxiter] = isnothing(maxiters) ? 1000 : maxiters
    da_kwargs[:initial_temp] = 5230.0
    da_kwargs[:restart_temp_ratio] = 2e-5
    da_kwargs[:visit] = 2.62
    da_kwargs[:accept] = -5.0
    da_kwargs[:maxfun] = 1e7
    da_kwargs[:no_local_search] = false
    
    if isa(cache.solver_args, NamedTuple)
        for (fname, fval) in pairs(cache.solver_args)
            if fname ∉ (:maxiters, :maxtime, :abstol, :reltol, :callback, :progress, :cons_tol)
                if !isnothing(fval)
                    da_kwargs[fname] = fval
                end
            end
        end
    end
    
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
            err_str = string(e.val)
            if occursin("Optimization halted by callback", err_str)
                throw(ErrorException("Optimization halted by callback"))
            else
                rethrow(e)
            end
        else
            rethrow(e)
        end
    end
    
    if isnothing(result)
        throw(ErrorException("Optimization failed to return a result"))
    end
    
    t1 = time()
    
    minimizer = pyconvert(Vector{Float64}, result.x)
    minimum = safe_to_float(result.fun)
    py_success = pyconvert(Bool, pybool(result.success))
    py_message = safe_get_message(result)
    
    if cache.sense === Optimization.MaxSense
        minimum = -minimum
    end
    
    retcode = py_success ? SciMLBase.ReturnCode.Success : SciMLBase.ReturnCode.Failure
    
    if retcode == SciMLBase.ReturnCode.Failure
        @warn "ScipyDualAnnealing failed to converge: $(py_message)"
    end
    
    stats = extract_stats(result, t1 - t0)
    
    return SciMLBase.build_solution(cache, cache.opt, minimizer, minimum;
                                   original = result,
                                   retcode = retcode,
                                   stats = stats)
end

function SciMLBase.__solve(cache::OptimizationCache{F,RC,LB,UB,LC,UC,S,O,D,P,C}) where 
                          {F,RC,LB,UB,LC,UC,S,O<:ScipyShgo,D,P,C}
    
    _loss = function(θ)
        θ_julia = ensure_julia_array(θ)
        x = cache.f(θ_julia, cache.p)
        opt_state = Optimization.OptimizationState(u = θ_julia, objective = x[1])
        if cache.callback(opt_state, x...)
            error("Optimization halted by callback")
        end
        return cache.sense === Optimization.MaxSense ? -x[1] : x[1]
    end
    
    bounds = pylist([pytuple([cache.lb[i], cache.ub[i]]) for i in 1:length(cache.lb)])
    
    constraints = nothing
    if !isnothing(cache.f.cons) && !isnothing(cache.lcons) && !isnothing(cache.ucons)
        cons_cache = zeros(Float64, length(cache.lcons))
        
        cons_list = []
        for i in 1:length(cache.lcons)
            if isfinite(cache.lcons[i])
                cons_func_lower = let i=i, cons_cache=cons_cache
                    function(θ)
                        θ_julia = ensure_julia_array(θ)
                        if hasmethod(cache.f.cons, Tuple{typeof(cons_cache), typeof(θ_julia), typeof(cache.p)})
                            cache.f.cons(cons_cache, θ_julia, cache.p)
                        else
                            cache.f.cons(cons_cache, θ_julia)
                        end
                        return cons_cache[i] - cache.lcons[i]
                    end
                end
                push!(cons_list, pydict(Dict("type" => "ineq", "fun" => cons_func_lower)))
            end
            
            if isfinite(cache.ucons[i])
                cons_func_upper = let i=i, cons_cache=cons_cache
                    function(θ)
                        θ_julia = ensure_julia_array(θ)
                        if hasmethod(cache.f.cons, Tuple{typeof(cons_cache), typeof(θ_julia), typeof(cache.p)})
                            cache.f.cons(cons_cache, θ_julia, cache.p)
                        else
                            cache.f.cons(cons_cache, θ_julia)
                        end
                        return cache.ucons[i] - cons_cache[i]
                    end
                end
                push!(cons_list, pydict(Dict("type" => "ineq", "fun" => cons_func_upper)))
            end
        end
        
        constraints = pylist(cons_list)
    end
    
    shgo_kwargs = Dict{Symbol, Any}()
    shgo_kwargs[:n] = 100
    shgo_kwargs[:iters] = 1
    shgo_kwargs[:sampling_method] = "simplicial"
    
    if isa(cache.solver_args, NamedTuple)
        for (fname, fval) in pairs(cache.solver_args)
            if fname ∉ (:maxiters, :maxtime, :abstol, :reltol, :callback, :progress, :cons_tol)
                if !isnothing(fval)
                    shgo_kwargs[fname] = fval
                end
            end
        end
    end
    
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
            err_str = string(e.val)
            if occursin("Optimization halted by callback", err_str)
                throw(ErrorException("Optimization halted by callback"))
            else
                rethrow(e)
            end
        else
            rethrow(e)
        end
    end
    
    if isnothing(result)
        throw(ErrorException("Optimization failed to return a result"))
    end
    
    t1 = time()
    
    minimizer = pyconvert(Vector{Float64}, result.x)
    minimum = safe_to_float(result.fun)
    py_success = pyconvert(Bool, pybool(result.success))
    py_message = safe_get_message(result)
    
    if cache.sense === Optimization.MaxSense
        minimum = -minimum
    end
    
    retcode = py_success ? SciMLBase.ReturnCode.Success : SciMLBase.ReturnCode.Failure
    
    if retcode == SciMLBase.ReturnCode.Failure
        @warn "ScipyShgo failed to converge: $(py_message)"
    end
    
    stats = extract_stats(result, t1 - t0)
    
    return SciMLBase.build_solution(cache, cache.opt, minimizer, minimum;
                                   original = result,
                                   retcode = retcode,
                                   stats = stats)
end

function SciMLBase.__solve(cache::OptimizationCache{F,RC,LB,UB,LC,UC,S,O,D,P,C}) where 
                          {F,RC,LB,UB,LC,UC,S,O<:ScipyDirect,D,P,C}
    
    _loss = function(θ)
        θ_julia = ensure_julia_array(θ)
        x = cache.f(θ_julia, cache.p)
        opt_state = Optimization.OptimizationState(u = θ_julia, objective = x[1])
        if cache.callback(opt_state, x...)
            error("Optimization halted by callback")
        end
        return cache.sense === Optimization.MaxSense ? -x[1] : x[1]
    end
    
    bounds = pylist([pytuple([cache.lb[i], cache.ub[i]]) for i in 1:length(cache.lb)])
    
    maxiters = Optimization._check_and_convert_maxiters(cache.solver_args.maxiters)
    
    direct_kwargs = Dict{Symbol, Any}()
    direct_kwargs[:eps] = 0.0001
    direct_kwargs[:maxiter] = isnothing(maxiters) ? 1000 : maxiters
    direct_kwargs[:locally_biased] = true
    direct_kwargs[:vol_tol] = 1e-16
    direct_kwargs[:len_tol] = 1e-6
    
    if isa(cache.solver_args, NamedTuple)
        for (fname, fval) in pairs(cache.solver_args)
            if fname ∉ (:maxiters, :maxtime, :abstol, :reltol, :callback, :progress, :cons_tol)
                if !isnothing(fval)
                    direct_kwargs[fname] = fval
                end
            end
        end
    end
    
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
            err_str = string(e.val)
            if occursin("Optimization halted by callback", err_str)
                throw(ErrorException("Optimization halted by callback"))
            else
                rethrow(e)
            end
        else
            rethrow(e)
        end
    end
    
    if isnothing(result)
        throw(ErrorException("Optimization failed to return a result"))
    end
    
    t1 = time()
    
    minimizer = pyconvert(Vector{Float64}, result.x)
    minimum = safe_to_float(result.fun)
    py_success = pyconvert(Bool, pybool(result.success))
    py_message = safe_get_message(result)
    
    if cache.sense === Optimization.MaxSense
        minimum = -minimum
    end
    
    retcode = py_success ? SciMLBase.ReturnCode.Success : SciMLBase.ReturnCode.Failure
    
    if retcode == SciMLBase.ReturnCode.Failure
        @warn "ScipyDirect failed to converge: $(py_message)"
    end
    
    stats = extract_stats(result, t1 - t0)
    
    return SciMLBase.build_solution(cache, cache.opt, minimizer, minimum;
                                   original = result,
                                   retcode = retcode,
                                   stats = stats)
end

function SciMLBase.__solve(cache::OptimizationCache{F,RC,LB,UB,LC,UC,S,O,D,P,C}) where 
                      {F,RC,LB,UB,LC,UC,S,O<:ScipyBrute,D,P,C}

    _loss = function(θ)
        θ_julia = ensure_julia_array(θ)
        x = cache.f(θ_julia, cache.p)
        opt_state = Optimization.OptimizationState(u = θ_julia, objective = x[1])
        if cache.callback(opt_state, x...)
            error("Optimization halted by callback")
        end
        return cache.sense === Optimization.MaxSense ? -x[1] : x[1]
    end

    ranges = pylist([pytuple([cache.lb[i], cache.ub[i]]) for i in 1:length(cache.lb)])

    brute_kwargs = Dict{Symbol, Any}()
    brute_kwargs[:Ns] = 20
    brute_kwargs[:full_output] = true
    brute_kwargs[:finish] = scipy.optimize.fmin
    brute_kwargs[:workers] = 1

    if isa(cache.solver_args, NamedTuple)
        for (fname, fval) in pairs(cache.solver_args)
            if fname ∉ (:maxiters, :maxtime, :abstol, :reltol, :callback, :progress, :cons_tol)
                if !isnothing(fval)
                    brute_kwargs[fname] = fval
                end
            end
        end
    end

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
            err_str = string(e.val)
            if occursin("Optimization halted by callback", err_str)
                throw(ErrorException("Optimization halted by callback"))
            else
                rethrow(e)
            end
        else
            rethrow(e)
        end
    end

    if isnothing(result)
        throw(ErrorException("Optimization failed to return a result"))
    end

    t1 = time()

    minimizer = pyconvert(Vector{Float64}, result[0])
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
   ScipyLBFGSB, ScipyTNC, ScipyCOBYLA, ScipySLSQP, ScipyTrustConstr,
   ScipyDogleg, ScipyTrustNCG, ScipyTrustKrylov, ScipyTrustExact,
   ScipyDifferentialEvolution, ScipyBasinhopping, ScipyDualAnnealing, 
   ScipyShgo, ScipyDirect, ScipyBrute

end # module