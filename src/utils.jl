struct NullData end
const DEFAULT_DATA = Iterators.cycle((NullData(),))
Base.iterate(::NullData, i=1) = nothing
Base.length(::NullData) = 0

get_maxiters(data) = Iterators.IteratorSize(typeof(DEFAULT_DATA)) isa Iterators.IsInfinite ||
                     Iterators.IteratorSize(typeof(DEFAULT_DATA)) isa Iterators.SizeUnknown ?
                     typemax(Int) : length(data)

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


function _map_optimizer_args(prob::OptimizationProblem, opt; kwargs...)
    __map_optimizer_args(prob, opt; kwargs...)
end

function _check_and_convert_maxiters(maxiters)
    if !(isnothing(maxiters)) && maxiters <= 0.0
        error("The number of maxiters has to be a non-negative and non-zero number.")
    elseif !(isnothing(maxiters))
        return convert(Int, maxiters)
    end
end

function _check_and_convert_maxtime(maxtime)
  if !(isnothing(maxtime)) && maxtime <= 0.0
      error("The maximum time has to be a non-negative and non-zero number.")
  elseif !(isnothing(maxtime))
      return convert(Float32, maxtime)
  end
end