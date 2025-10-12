module OptimizationBase

using DocStringExtensions
using Reexport
@reexport using SciMLBase, ADTypes

using ArrayInterface, Base.Iterators, SparseArrays, LinearAlgebra
import SciMLBase: OptimizationProblem,
                  OptimizationFunction, ObjSense,
                  MaxSense, MinSense, OptimizationStats,
                  allowsbounds, requiresbounds,
                  allowsconstraints, requiresconstraints,
                  allowscallback, requiresgradient,
                  requireshessian, requiresconsjac,
                  requiresconshess, supports_opt_cache_interface
export ObjSense, MaxSense, MinSense
export allowsbounds, requiresbounds, allowsconstraints, requiresconstraints,
       allowscallback, requiresgradient, requireshessian,
       requiresconsjac, requiresconshess, supports_opt_cache_interface

using FastClosures

struct NullCallback end
(x::NullCallback)(args...) = false
const DEFAULT_CALLBACK = NullCallback()

struct NullData end
const DEFAULT_DATA = Iterators.cycle((NullData(),))
Base.iterate(::NullData, i = 1) = nothing
Base.length(::NullData) = 0

include("adtypes.jl")
include("symify.jl")
include("cache.jl")
include("solve.jl")
include("OptimizationDIExt.jl")
include("OptimizationDISparseExt.jl")
include("function.jl")
include("utils.jl")
include("state.jl")

export solve, OptimizationCache, DEFAULT_CALLBACK, DEFAULT_DATA
export IncompatibleOptimizerError, OptimizerMissingError

end
