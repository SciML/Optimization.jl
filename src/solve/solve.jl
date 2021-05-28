struct NullData end
const DEFAULT_DATA = Iterators.cycle((NullData(),))
Base.iterate(::NullData, i=1) = nothing
Base.length(::NullData) = 0

get_maxiters(data) = Iterators.IteratorSize(typeof(DEFAULT_DATA)) isa Iterators.IsInfinite ||
                     Iterators.IteratorSize(typeof(DEFAULT_DATA)) isa Iterators.SizeUnknown ?
                     typemax(Int) : length(data)

#=
function update!(x::AbstractArray, x̄::AbstractArray{<:ForwardDiff.Dual})
  x .-= x̄
end

function update!(x::AbstractArray, x̄)
  x .-= getindex.(ForwardDiff.partials.(x̄),1)
end

function update!(opt, x, x̄)
  x .-= Flux.Optimise.apply!(opt, x, x̄)
end

function update!(opt, x, x̄::AbstractArray{<:ForwardDiff.Dual})
  x .-= Flux.Optimise.apply!(opt, x, getindex.(ForwardDiff.partials.(x̄),1))
end

function update!(opt, xs::Flux.Zygote.Params, gs)
    update!(opt, xs[1], gs)
end
=#

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
