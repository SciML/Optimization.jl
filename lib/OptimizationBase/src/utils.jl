function get_maxiters(data)
    return Iterators.IteratorSize(typeof(DEFAULT_DATA)) isa Iterators.IsInfinite ||
        Iterators.IteratorSize(typeof(DEFAULT_DATA)) isa Iterators.SizeUnknown ?
        typemax(Int) : length(data)
end

# Sense handling
supports_sense(::Any) = false

function apply_sense(f::OptimizationFunction{iip}, sense) where {iip}
    if sense == SciMLBase.MinSense
        return f
    end

    objf = (u, p) -> -f.f(u, p)

    grad = if f.grad === nothing
        nothing
    elseif iip
        function (G, u, p)
            f.grad(G, u, p)
            G .*= -one(eltype(G))
        end
    else
        (u, p) -> -f.grad(u, p)
    end

    fg = if f.fg === nothing
        nothing
    elseif iip
        function (G, u, p)
            y = f.fg(G, u, p)
            G .*= -one(eltype(G))
            return -y
        end
    else
        function (u, p)
            y, g = f.fg(u, p)
            return -y, -g
        end
    end

    hess = if f.hess === nothing
        nothing
    elseif iip
        function (H, u, p)
            f.hess(H, u, p)
            H .*= -one(eltype(H))
        end
    else
        (u, p) -> -f.hess(u, p)
    end

    fgh = if f.fgh === nothing
        nothing
    elseif iip
        function (G, H, u, p)
            y = f.fgh(G, H, u, p)
            G .*= -one(eltype(G))
            H .*= -one(eltype(H))
            return -y
        end
    else
        function (u, p)
            y, g, h = f.fgh(u, p)
            return -y, -g, -h
        end
    end

    hv = if f.hv === nothing
        nothing
    elseif iip
        function (Hv, u, v, p)
            f.hv(Hv, u, v, p)
            Hv .*= -one(eltype(Hv))
        end
    else
        (u, v, p) -> -f.hv(u, v, p)
    end

    lag_h = if f.lag_h === nothing
        nothing
    elseif iip
        (H, u, σ, μ, p) -> f.lag_h(H, u, -σ, μ, p)
    else
        (u, σ, μ, p) -> f.lag_h(u, -σ, μ, p)
    end

    return OptimizationFunction{iip}(
        objf, f.adtype;
        grad = grad, fg = fg, hess = hess, hv = hv, fgh = fgh,
        cons = f.cons, cons_j = f.cons_j, cons_jvp = f.cons_jvp,
        cons_vjp = f.cons_vjp, cons_h = f.cons_h,
        hess_prototype = f.hess_prototype,
        cons_jac_prototype = f.cons_jac_prototype,
        cons_hess_prototype = f.cons_hess_prototype,
        observed = f.observed,
        expr = f.expr,
        cons_expr = f.cons_expr,
        sys = f.sys,
        lag_h = lag_h,
        lag_hess_prototype = f.lag_hess_prototype,
        hess_colorvec = f.hess_colorvec,
        cons_jac_colorvec = f.cons_jac_colorvec,
        cons_hess_colorvec = f.cons_hess_colorvec,
        lag_hess_colorvec = f.lag_hess_colorvec,
        initialization_data = f.initialization_data
    )
end

apply_sense(f::MultiObjectiveOptimizationFunction, sense) = f

decompose_trace(trace) = trace

function _check_and_convert_maxiters(maxiters)
    if !(isnothing(maxiters)) && maxiters <= 0.0
        error("The number of maxiters has to be a non-negative and non-zero number.")
    elseif !(isnothing(maxiters))
        return convert(Int, round(maxiters))
    end
end

function _check_and_convert_maxtime(maxtime)
    if !(isnothing(maxtime)) && maxtime <= 0.0
        error("The maximum time has to be a non-negative and non-zero number.")
    elseif !(isnothing(maxtime))
        return convert(Float32, maxtime)
    end
end

# RetCode handling for BBO and others.
using SciMLBase: ReturnCode

# Define a dictionary to map regular expressions to ReturnCode values
const STOP_REASON_MAP = Dict(
    r"Delta fitness .* below tolerance .*" => ReturnCode.Success,
    r"Fitness .* within tolerance .* of optimum" => ReturnCode.Success,
    r"CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL" => ReturnCode.Success,
    r"^CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR\*EPSMCH\s*$" => ReturnCode.Success,
    r"Terminated" => ReturnCode.Terminated,
    r"MaxIters|MAXITERS_EXCEED|Max number of steps .* reached" => ReturnCode.MaxIters,
    r"MaxTime|TIME_LIMIT" => ReturnCode.MaxTime,
    r"Max time" => ReturnCode.MaxTime,
    r"DtLessThanMin" => ReturnCode.DtLessThanMin,
    r"Unstable" => ReturnCode.Unstable,
    r"InitialFailure" => ReturnCode.InitialFailure,
    r"ConvergenceFailure|ITERATION_LIMIT" => ReturnCode.ConvergenceFailure,
    r"Infeasible|INFEASIBLE|DUAL_INFEASIBLE|LOCALLY_INFEASIBLE|INFEASIBLE_OR_UNBOUNDED" => ReturnCode.Infeasible,
    r"TOTAL NO. of ITERATIONS REACHED LIMIT" => ReturnCode.MaxIters,
    r"TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT" => ReturnCode.MaxIters,
    r"ABNORMAL_TERMINATION_IN_LNSRCH" => ReturnCode.Unstable,
    r"ERROR INPUT DATA" => ReturnCode.InitialFailure,
    r"FTOL.TOO.SMALL" => ReturnCode.ConvergenceFailure,
    r"GTOL.TOO.SMALL" => ReturnCode.ConvergenceFailure,
    r"XTOL.TOO.SMALL" => ReturnCode.ConvergenceFailure,
    r"STOP: TERMINATION" => ReturnCode.Terminated,
    r"Optimization completed" => ReturnCode.Success,
    r"Convergence achieved" => ReturnCode.Success,
    r"ROUNDOFF_LIMITED" => ReturnCode.Success
)

# Function to deduce ReturnCode from a stop_reason string using the dictionary
function deduce_retcode(stop_reason::String, verbose = OptimizationVerbosity())
    for (pattern, retcode) in STOP_REASON_MAP
        if occursin(pattern, stop_reason)
            return retcode
        end
    end

    @SciMLMessage(
        lazy"Unrecognized stop reason: $stop_reason. Defaulting to ReturnCode.Default.",
        verbose, :unrecognized_stop_reason
    )

    return ReturnCode.Default
end

# Function to deduce ReturnCode from a Symbol
function deduce_retcode(retcode::Symbol)
    if retcode == :Default || retcode == :DEFAULT
        return ReturnCode.Default
    elseif retcode == :Success || retcode == :EXACT_SOLUTION_LEFT ||
            retcode == :FLOATING_POINT_LIMIT || retcode == :true || retcode == :OPTIMAL ||
            retcode == :LOCALLY_SOLVED || retcode == :ROUNDOFF_LIMITED ||
            retcode == :SUCCESS ||
            retcode == :STOPVAL_REACHED || retcode == :FTOL_REACHED ||
            retcode == :XTOL_REACHED
        return ReturnCode.Success
    elseif retcode == :Terminated
        return ReturnCode.Terminated
    elseif retcode == :MaxIters || retcode == :MAXITERS_EXCEED ||
            retcode == :MAXEVAL_REACHED
        return ReturnCode.MaxIters
    elseif retcode == :MaxTime || retcode == :TIME_LIMIT || retcode == :MAXTIME_REACHED
        return ReturnCode.MaxTime
    elseif retcode == :DtLessThanMin
        return ReturnCode.DtLessThanMin
    elseif retcode == :Unstable
        return ReturnCode.Unstable
    elseif retcode == :InitialFailure
        return ReturnCode.InitialFailure
    elseif retcode == :ConvergenceFailure || retcode == :ITERATION_LIMIT
        return ReturnCode.ConvergenceFailure
    elseif retcode == :Failure || retcode == :false
        return ReturnCode.Failure
    elseif retcode == :Infeasible || retcode == :INFEASIBLE ||
            retcode == :DUAL_INFEASIBLE || retcode == :LOCALLY_INFEASIBLE ||
            retcode == :INFEASIBLE_OR_UNBOUNDED
        return ReturnCode.Infeasible
    else
        return ReturnCode.Failure
    end
end

function SciMLBase.build_solution(cache::OptimizationCache, alg, u, objective; kwargs...)
    if cache.sense === MaxSense && !supports_sense(cache.opt)
        objective = -objective
    end
    return invoke(
        SciMLBase.build_solution,
        Tuple{SciMLBase.AbstractOptimizationCache, typeof(alg), typeof(u), typeof(objective)},
        cache, alg, u, objective; kwargs...
    )
end
