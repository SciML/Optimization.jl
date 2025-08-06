module OptimizationBase

using DocStringExtensions
using Reexport
@reexport using SciMLBase, ADTypes

using ArrayInterface, Base.Iterators, SparseArrays, LinearAlgebra
import SciMLBase: OptimizationProblem,
                  OptimizationFunction, ObjSense,
                  MaxSense, MinSense, OptimizationStats
export ObjSense, MaxSense, MinSense

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

export solve, OptimizationCache, DEFAULT_CALLBACK, DEFAULT_DATA

end
