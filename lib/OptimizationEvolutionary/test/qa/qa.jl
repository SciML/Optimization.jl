using SciMLTesting, OptimizationEvolutionary, JET
using Test
using Evolutionary

# ExplicitImports findings, all tracked against SciML/Optimization.jl:
#  * no_implicit_imports broken: the module relies on `@reexport`/`using`
#    module names (SciMLBase/OptimizationBase/Reexport/...) that cannot be made
#    explicit without restructuring.
#  * the ignored *_are_public / *_via_owners names are owned by SciMLBase,
#    OptimizationBase, the backend, or Base and are not (yet) declared public;
#    the proper fix is upstream `public` declarations, not a local change.
# SciML trait/interface methods are our own, not piracy — mark them as such.
# The Evolutionary.trace! override IS genuine piracy (changes Evolutionary's
# tracing globally); mark the piracy test broken until it's replaced.
SB = OptimizationEvolutionary.SciMLBase
run_qa(
    OptimizationEvolutionary;
    explicit_imports = true,
    aqua_kwargs = (;
        piracies = (;
            broken = true,
            treat_as_own = [
                SB.__solve,
                SB.allowsbounds,
                SB.allowscallback,
                SB.allowsconstraints,
                SB.has_init,
                SB.requiresconshess,
                SB.requiresconsjac,
                SB.requiresgradient,
                SB.requireshessian,
            ],
        ),
    ),
    ei_kwargs = (;
        all_qualified_accesses_via_owners = (; ignore = (:OptimizationStats, :minimum)),
        all_qualified_accesses_are_public = (; ignore = (:AbstractOptimizer, :OptimizationState, :OptimizationStats, :OptimizationTrace, :OptimizationTraceRecord, :Options, :__solve, :_check_and_convert_maxiters, :_check_and_convert_maxtime, :allowscallback, :converged, :minimizer, :minimum, :optimize, :requiresconshess, :requiresconsjac, :requiresgradient, :requireshessian, :trace!, :update!)),
    ),
    api_docs_kwargs = (;
        ignore = (
            :AutoModelingToolkit,
            :AutoSparseFastDifferentiation,
            :AutoSparseFiniteDiff,
            :AutoSparseForwardDiff,
            :AutoSparsePolyesterForwardDiff,
            :AutoSparseReverseDiff,
            :AutoSparseZygote,
            :Terminal,
            :default_options,
            :discrete,
            :domainrange,
            :exponential,
            :intermediate,
            :line,
            :mutationwrapper,
            :singlepoint,
            :strategy,
            :twopoint,
            :uniformbin,
            :waverage,
        ),
    ),
    ei_broken = (:no_implicit_imports,),
)
