module OptimizationBase

using DocStringExtensions
using Reexport
@reexport using SciMLBase, ADTypes, SciMLLogging

using ArrayInterface, Base.Iterators, SparseArrays, LinearAlgebra
import SciMLBase: solve, init, solve!, __init, __solve,
    OptimizationProblem,
    OptimizationFunction, ObjSense,
    MaxSense, MinSense, OptimizationStats,
    allowsbounds, requiresbounds,
    allowsconstraints, requiresconstraints,
    allowscallback, requiresgradient,
    requireshessian, requiresconsjac,
    requiresconshess, wrap_sol, has_kwargs,
    get_root_indp, get_updated_symbolic_problem,
    get_concrete_p, get_concrete_u0, promote_u0,
    KeywordArgError, extract_alg,
    _concrete_solve_adjoint, _concrete_solve_forward

using SymbolicIndexingInterface: SymbolicIndexingInterface

export ObjSense, MaxSense, MinSense
export allowsbounds, requiresbounds, allowsconstraints, requiresconstraints,
    allowscallback, requiresgradient, requireshessian,
    requiresconsjac, requiresconshess

using FastClosures

struct NullCallback end
(x::NullCallback)(args...) = false
const DEFAULT_CALLBACK = NullCallback()

struct NullData end
const DEFAULT_DATA = Iterators.cycle((NullData(),))
Base.iterate(::NullData, i = 1) = nothing
Base.length(::NullData) = 0

include("verbosity.jl")
include("solve.jl")
include("adtypes.jl")
include("symify.jl")
include("cache.jl")
include("OptimizationDIExt.jl")
include("OptimizationDISparseExt.jl")
include("function.jl")
include("utils.jl")
include("state.jl")

export solve, OptimizationCache, DEFAULT_CALLBACK, DEFAULT_DATA
export IncompatibleOptimizerError, OptimizerMissingError
export OptimizationVerbosity

include("precompilation.jl")

end
