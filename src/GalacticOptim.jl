"""
$(DocStringExtensions.README)
"""
module GalacticOptim

using DocStringExtensions
using Reexport
@reexport using DiffEqBase
@reexport using SciMLBase
using Requires
using DiffResults, ForwardDiff, Zygote, ReverseDiff, Tracker, FiniteDiff
@reexport using Optim, Flux
using Logging, ProgressLogging, Printf, ConsoleProgressMonitor, TerminalLoggers, LoggingExtras
using ArrayInterface, Base.Iterators

using ForwardDiff: DEFAULT_CHUNK_THRESHOLD
import SciMLBase: OptimizationProblem, OptimizationFunction, AbstractADType, __solve

import ModelingToolkit
import ModelingToolkit: AutoModelingToolkit
export AutoModelingToolkit

include("solve.jl")
include("function.jl")

export solve

export BBO, CMAEvolutionStrategyOpt

end # module
