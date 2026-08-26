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
    :EnsembleDistributed, :EnsembleProblem, :EnsembleSerial, :EnsembleThreads,
    :MaxSense, :MinSense, :NoAD, :ObjSense,
    :MultiObjectiveOptimizationFunction, :OptimizationFunction, :OptimizationProblem,
    :OptimizationSolution, :OptimizationStats,
    :ReturnCode, :allowsbounds, :allowscallback, :allowsconstraints,
    :init, :remake, :requiresbounds, :requiresconshess, :requiresconsjac,
    :requiresconstraints, :requiresgradient, :requireshessian, :solve, :solve!,
    :successful_retcode,
)

# A package's re-surfaced names must actually be exported *and* resolvable from
# `using <Pkg>`; asserting both keeps the `reexports_allow` list from drifting away
# from the `export` block in src/ (the drift is what silently strips public API).
function test_reexported_names(mod::Module, reexports)
    exported = names(mod)
    for name in reexports
        @test name in exported
        @test isdefined(mod, name)
    end
    return nothing
end
