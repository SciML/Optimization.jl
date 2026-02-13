"""
$(DocStringExtensions.README)
"""
module Optimization

using DocStringExtensions
using Reexport
@reexport using SciMLBase, ADTypes, OptimizationBase

using Logging, ConsoleProgressMonitor, TerminalLoggers, LoggingExtras
using ArrayInterface, Base.Iterators, SparseArrays, LinearAlgebra

import OptimizationBase: instantiate_function, OptimizationCache, ReInitCache
import SciMLBase: OptimizationProblem,
    OptimizationFunction, ObjSense,
    MaxSense, MinSense, OptimizationStats
export ObjSense, MaxSense, MinSense

export solve

end # module
