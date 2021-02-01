module GalacticOptim

using Reexport
@reexport using DiffEqBase
using Requires
using DiffResults, ForwardDiff, Zygote, ReverseDiff, Tracker, FiniteDiff
@reexport using Optim, Flux
using Logging, ProgressLogging, Printf, ConsoleProgressMonitor, TerminalLoggers, LoggingExtras
using ArrayInterface, Base.Iterators

using ForwardDiff: DEFAULT_CHUNK_THRESHOLD
import DiffEqBase: OptimizationProblem, OptimizationFunction, AbstractADType

include("solve.jl")
include("function.jl")

export solve, EnsembleOptimizationProblem

export BBO, CMAEvolutionStrategyOpt

end # module
