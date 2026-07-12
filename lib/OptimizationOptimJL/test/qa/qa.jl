using SciMLTesting, OptimizationOptimJL, JET, OptimizationBase, SciMLBase
using Test
using Optim

# ExplicitImports findings, all tracked against SciML/Optimization.jl:
#  * no_implicit_imports broken: the module relies on `@reexport`/`using`
#    module names (SciMLBase/OptimizationBase/Reexport/...) that cannot be made
#    explicit without restructuring.
#  * the ignored *_are_public / *_via_owners names are owned by SciMLBase,
#    OptimizationBase, the backend, or Base and are not (yet) declared public;
#    the proper fix is upstream `public` declarations, not a local change.
# OptimizationOptimJL implements the SciML optimization interface for Optim,
# so the trait/interface methods it adds extend SciML's *own* functions rather
# than committing type piracy — mark those functions as own.
run_qa(
    OptimizationOptimJL;
    explicit_imports = true,
    aqua_kwargs = (;
        piracies = (;
            treat_as_own = [
                SciMLBase.__init,
                SciMLBase.__solve,
                SciMLBase.allowsbounds,
                SciMLBase.allowscallback,
                SciMLBase.allowsconstraints,
                SciMLBase.allowsfg,
                SciMLBase.has_init,
                SciMLBase.requiresbounds,
                SciMLBase.requiresconshess,
                SciMLBase.requiresconsjac,
                SciMLBase.requiresgradient,
                SciMLBase.requireshessian,
                OptimizationBase.supports_sense,
            ],
        ),
    ),
    ei_kwargs = (;
        all_qualified_accesses_via_owners = (; ignore = (:NLSolversBase, :OptimizationStats)),
        all_qualified_accesses_are_public = (; ignore = (:AbstractConstrainedOptimizer, :AbstractOptimizer, :ConstrainedOptimizer, :KrylovTrustRegion, :NLSolversBase, :NoAD, :OptimizationState, :OptimizationStats, :Options, :ZerothOrderOptimizer, :__init, :__solve, :_check_and_convert_maxiters, :_check_and_convert_maxtime, :alloc_DF, :alloc_H, :allowscallback, :allowsfg, :converged, :iteration_limit_reached, :requiresbounds, :requiresconshess, :requiresconsjac, :requiresgradient, :requireshessian, :supports_sense, :value)),
    ),
    api_docs_kwargs = (;
        ignore = (
            :AcceleratedGradientDescent,
            :AutoModelingToolkit,
            :AutoSparseFastDifferentiation,
            :AutoSparseFiniteDiff,
            :AutoSparseForwardDiff,
            :AutoSparsePolyesterForwardDiff,
            :AutoSparseReverseDiff,
            :AutoSparseZygote,
            :Manifold,
            :MomentumGradientDescent,
            :NonDifferentiable,
            :OnceDifferentiable,
            :OptimizationState,
            :OptimizationTrace,
            :TwiceDifferentiable,
            :TwiceDifferentiableConstraints,
            :initial_state,
            :maximize,
            :optimize,
        ),
    ),
    ei_broken = (:no_implicit_imports,),
)
