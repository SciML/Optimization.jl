"""
    OptimizationBase

Core types, defaults, and solver interface extensions shared by the
Optimization.jl solver packages.
"""
module OptimizationBase

using DocStringExtensions
# Deliberately NOT `@reexport`: blanket-reexporting these put 256 names into every
# downstream namespace — `DynamicalSDEFunction`, `BlockDiagonalOperator`, `AddVector` and
# the rest of SciMLOperators — of which the whole monorepo referenced 65. The public
# surface is the explicit `export` lists below; anything else is reached through its
# owner (`SciMLBase.x`, `ADTypes.x`, `SciMLLogging.x`).
using SciMLBase, ADTypes, SciMLLogging

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

@doc """
    ObjSense

Abstract objective-sense marker used by `OptimizationProblem` to indicate
whether the objective should be minimized or maximized.

See also [`MinSense`](@ref) and [`MaxSense`](@ref).
""" ObjSense

@doc """
    MinSense

Objective sense for minimizing an `OptimizationProblem`.
""" MinSense

@doc """
    MaxSense

Objective sense for maximizing an `OptimizationProblem`.
""" MaxSense

using SymbolicIndexingInterface: SymbolicIndexingInterface

export ObjSense, MaxSense, MinSense
export allowsbounds, requiresbounds, allowsconstraints, requiresconstraints,
    allowscallback, requiresgradient, requireshessian,
    requiresconsjac, requiresconshess

using FastClosures

struct NullCallback end
(x::NullCallback)(args...) = false

"""
    DEFAULT_CALLBACK

Default callback for `solve` and `init`. It ignores all callback arguments and
returns `false`, so optimization continues until the solver stops.
"""
const DEFAULT_CALLBACK = NullCallback()

struct NullData end

"""
    DEFAULT_DATA

Default data iterator for optimization problems that are not minibatched.
"""
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

# Intentionally re-surfaced API owned elsewhere. Kept to what a user actually needs to
# state and solve an optimization problem; `run_qa`'s `reexports_allow` lists exactly
# these, so adding one is a deliberate, reviewable act.
export OptimizationProblem, OptimizationFunction, MultiObjectiveOptimizationFunction,
    OptimizationSolution, OptimizationStats
export init, solve!, remake, ReturnCode
export AutoEnzyme, AutoFiniteDiff, AutoForwardDiff, AutoMooncake, AutoReverseDiff,
    AutoSparse, AutoSymbolics, AutoTracker, AutoZygote

include("precompilation.jl")

end
