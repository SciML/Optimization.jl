"""
$(DocStringExtensions.README)
"""
module Optimization

using DocStringExtensions
using Reexport
# `OptimizationBase` curates its own public surface, so re-exporting it wholesale is a
# bounded, reviewable act. `SciMLBase`/`ADTypes` are deliberately *not* re-exported:
# blanket-reexporting them is what put 263 names — most of SciMLOperators among them —
# into every `using Optimization` namespace.
@reexport using OptimizationBase
using SciMLBase, ADTypes

using Logging, ConsoleProgressMonitor, TerminalLoggers, LoggingExtras
using ArrayInterface, Base.Iterators, SparseArrays, LinearAlgebra

import OptimizationBase: instantiate_function, OptimizationCache, ReInitCache
import SciMLBase: OptimizationProblem, OptimizationFunction, ObjSense,
    MaxSense, MinSense, OptimizationStats, OptimizationSolution

@doc """
    ObjSense

Objective-sense marker used by `OptimizationProblem` to select minimization or
maximization.
""" ObjSense

@doc """
    MinSense

Objective sense for minimizing an `OptimizationProblem`.
""" MinSense

@doc """
    MaxSense

Objective sense for maximizing an `OptimizationProblem`.
""" MaxSense

@doc """
    OptimizationSolution

Solution type returned by an optimization solver. The concrete fields and
solution interface are defined by SciMLBase.
""" OptimizationSolution

export ObjSense, MaxSense, MinSense
export solve

end # module
