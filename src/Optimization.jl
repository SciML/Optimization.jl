"""
$(DocStringExtensions.README)
"""
module Optimization

import DocStringExtensions: DocStringExtensions
import Reexport: Reexport, @reexport
@reexport import OptimizationBase: OptimizationBase, AutoEnzyme, AutoFiniteDiff,
    AutoForwardDiff,
    AutoMooncake, AutoReverseDiff, AutoSparse, AutoSymbolics, AutoTracker, AutoZygote,
    DEFAULT_CALLBACK, DEFAULT_DATA, EnsembleDistributed, EnsembleProblem, EnsembleSerial,
    EnsembleThreads, IncompatibleOptimizerError, MaxSense, MinSense,
    MultiObjectiveOptimizationFunction, NoAD, ObjSense, OptimizationCache,
    OptimizationFunction, OptimizationProblem, OptimizationSolution, OptimizationStats,
    OptimizationVerbosity, OptimizerMissingError, ReturnCode, allowsbounds, allowscallback,
    allowsconstraints, init, lag_hess_structure, remake, requiresbounds, requiresconshess,
    requiresconsjac, requiresconstraints, requiresgradient, requireshessian, solve, solve!,
    successful_retcode
import SciMLBase: SciMLBase
import ADTypes: ADTypes

import Logging: Logging
import ConsoleProgressMonitor: ConsoleProgressMonitor
import TerminalLoggers: TerminalLoggers
import LoggingExtras: LoggingExtras
import ArrayInterface: ArrayInterface
import Base.Iterators: Iterators
import SparseArrays: SparseArrays
import LinearAlgebra: LinearAlgebra

import OptimizationBase: instantiate_function, OptimizationCache, ReInitCache
import SciMLBase: OptimizationProblem, OptimizationFunction, OptimizationStats

export OptimizationBase

end # module
