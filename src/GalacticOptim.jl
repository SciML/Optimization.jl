module GalacticOptim

using DiffEqBase, Requires
using DiffResults, ForwardDiff, Zygote, ReverseDiff, Tracker, FiniteDiff
using Optim, Flux
using Logging, ProgressLogging, Printf, ConsoleProgressMonitor, TerminalLoggers, LoggingExtras

import DiffEqBase: OptimizationProblem

include("problem.jl")
include("solve.jl")
include("function.jl")

export OptimizationFunction
export solve

export BBO, CMAEvolutionStrategyOpt

end # module
