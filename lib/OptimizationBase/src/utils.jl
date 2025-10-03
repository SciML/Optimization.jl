function get_maxiters(data)
    Iterators.IteratorSize(typeof(DEFAULT_DATA)) isa Iterators.IsInfinite ||
        Iterators.IteratorSize(typeof(DEFAULT_DATA)) isa Iterators.SizeUnknown ?
    typemax(Int) : length(data)
end

maybe_with_logger(f, logger) = logger === nothing ? f() : Logging.with_logger(f, logger)

function default_logger(logger)
    Logging.min_enabled_level(logger) ≤ ProgressLogging.ProgressLevel && return nothing
    if Sys.iswindows() || (isdefined(Main, :IJulia) && Main.IJulia.inited)
        progresslogger = ConsoleProgressMonitor.ProgressLogger()
    else
        progresslogger = TerminalLoggers.TerminalLogger()
    end
    logger1 = LoggingExtras.EarlyFilteredLogger(progresslogger) do log
        log.level == ProgressLogging.ProgressLevel
    end
    logger2 = LoggingExtras.EarlyFilteredLogger(logger) do log
        log.level != ProgressLogging.ProgressLevel
    end
    LoggingExtras.TeeLogger(logger1, logger2)
end

macro withprogress(progress, exprs...)
    quote
        if $progress
            $maybe_with_logger($default_logger($Logging.current_logger())) do
                $ProgressLogging.@withprogress $(exprs...)
            end
        else
            $(exprs[end])
        end
    end |> esc
end

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
function deduce_retcode(stop_reason::String)
    for (pattern, retcode) in STOP_REASON_MAP
        if occursin(pattern, stop_reason)
            return retcode
        end
    end
    @warn "Unrecognized stop reason: $stop_reason. Defaulting to ReturnCode.Default."
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
