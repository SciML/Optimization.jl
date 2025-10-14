module OptimizationBase

using DocStringExtensions
using Reexport
@reexport using SciMLBase, ADTypes

using ArrayInterface, Base.Iterators, SparseArrays, LinearAlgebra
import SciMLBase: solve, init, solve!, __init, __solve,
                  OptimizationProblem,
                  OptimizationFunction, ObjSense,
                  MaxSense, MinSense, OptimizationStats,
                  allowsbounds, requiresbounds,
                  allowsconstraints, requiresconstraints,
                  allowscallback, requiresgradient,
                  requireshessian, requiresconsjac,
                  requiresconshess

@static if isdefined(SciMLBase, :supports_opt_cache_interface)
    import SciMLBase: supports_opt_cache_interface
else
    supports_opt_cache_interface(alg) = false
end

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
include("OptimizationDIExt.jl")
include("OptimizationDISparseExt.jl")
include("function.jl")
include("solve.jl")
include("utils.jl")
include("state.jl")

export solve, OptimizationCache, DEFAULT_CALLBACK, DEFAULT_DATA,
       IncompatibleOptimizerError, OptimizerMissingError, _check_opt_alg,
       supports_opt_cache_interface

end
