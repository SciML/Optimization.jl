"""
$(DocStringExtensions.README)
"""
module GalacticOptim

using DocStringExtensions
using Reexport
@reexport using SciMLBase
using Requires
using DiffResults, ForwardDiff, Zygote, ReverseDiff, Tracker, FiniteDiff
using Logging, ProgressLogging, Printf, ConsoleProgressMonitor, TerminalLoggers, LoggingExtras
using ArrayInterface, Base.Iterators

using ForwardDiff: DEFAULT_CHUNK_THRESHOLD
import SciMLBase: OptimizationProblem, OptimizationFunction, AbstractADType, __solve

include("solve.jl")
include("function.jl")

export solve

export BBO, CMAEvolutionStrategyOpt

end # module
