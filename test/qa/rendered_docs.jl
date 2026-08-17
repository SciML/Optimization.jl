using SciMLTesting: public_api_names

# `run_qa`'s reexport audit flags every public name a package does not own. The glue
# packages deliberately surface their backend's solver names — that is what they are for
# — so allow the public API of the modules each one still `@reexport`s. Everything else
# must be owned; the umbrella packages instead list their curated re-surface explicitly.
function optimization_reexports_allow(mods::Module...)
    allowed = Symbol[]
    for m in mods
        push!(allowed, nameof(m))
        append!(allowed, public_api_names(m))
    end
    return Tuple(sort!(unique!(allowed)))
end

# The names Optimization/OptimizationBase intentionally re-surface from SciMLBase and
# ADTypes so that `using Optimization` is enough to state and solve a problem. Adding to
# this list is a deliberate widening of the public API, not an accident of `@reexport`.
const OPTIMIZATION_CURATED_REEXPORTS = (
    :AutoEnzyme, :AutoFiniteDiff, :AutoForwardDiff, :AutoMooncake, :AutoReverseDiff,
    :AutoSparse, :AutoSymbolics, :AutoTracker, :AutoZygote,
    :MaxSense, :MinSense, :ObjSense,
    :MultiObjectiveOptimizationFunction, :OptimizationFunction, :OptimizationProblem,
    :OptimizationSolution, :OptimizationStats,
    :ReturnCode, :allowsbounds, :allowscallback, :allowsconstraints,
    :init, :remake, :requiresbounds, :requiresconshess, :requiresconsjac,
    :requiresconstraints, :requiresgradient, :requireshessian, :solve, :solve!,
)
