using SciMLTesting, OptimizationMOI, JET, OptimizationBase, SciMLBase
using Test

include(normpath(joinpath(@__DIR__, "..", "..", "..", "..", "test", "qa", "rendered_docs.jl")))

using MathOptInterface

# ExplicitImports findings, all tracked against SciML/Optimization.jl:
#  * no_implicit_imports broken: the module relies on `@reexport`/`using`
#    module names (SciMLBase/OptimizationBase/Reexport/...) that cannot be made
#    explicit without restructuring.
#  * the ignored *_are_public / *_via_owners names are owned by SciMLBase,
#    OptimizationBase, the backend, or Base and are not (yet) declared public;
#    the proper fix is upstream `public` declarations, not a local change.
#  * the ignored stale imports are part of the intentionally re-surfaced API.
# OptimizationMOI implements the SciML optimization interface for
# MathOptInterface, so the trait/interface methods it adds extend SciML's *own*
# functions rather than committing type piracy — mark those functions as own.
run_qa(
    OptimizationMOI;
    explicit_imports = true,
    aqua_kwargs = (;
        piracies = (;
            treat_as_own = [
                SciMLBase.__init,
                SciMLBase.__solve,
                SciMLBase.allowsbounds,
                SciMLBase.allowscallback,
                SciMLBase.allowsconstraints,
                SciMLBase.get_observed,
                SciMLBase.get_p,
                SciMLBase.get_paramsyms,
                SciMLBase.get_syms,
                SciMLBase.has_init,
                SciMLBase.requiresconshess,
                SciMLBase.requiresconsjac,
                SciMLBase.requiresgradient,
                SciMLBase.requireshessian,
                SciMLBase.supports_opt_cache_interface,
                OptimizationBase.supports_sense,
            ],
        ),
    ),
    ei_kwargs = (;
        no_stale_explicit_imports = (; ignore = (:parameters, :toexpr, :unknowns, :varmap_to_vars)),
        all_qualified_accesses_via_owners = (; ignore = (Symbol("@sync"), :OptimizationStats)),
        all_qualified_accesses_are_public = (; ignore = (Symbol("@sync"), :ALMOST_DUAL_INFEASIBLE, :ALMOST_INFEASIBLE, :ALMOST_LOCALLY_SOLVED, :ALMOST_OPTIMAL, :AbstractNLPEvaluator, :AbstractOptimizationCache, :AbstractOptimizer, :BarrierIterations, :CachingOptimizer, :Code, :DEFAULT_VERBOSE, :DUAL_INFEASIBLE, :EqualTo, :GetAttributeNotAllowed, :GreaterThan, :INFEASIBLE, :INFEASIBLE_OR_UNBOUNDED, :INTERRUPTED, :INVALID_MODEL, :INVALID_OPTION, :ITERATION_LIMIT, :Integer, :LOCALLY_INFEASIBLE, :LOCALLY_SOLVED, :LessThan, :MAX_SENSE, :MEMORY_LIMIT, :MIN_SENSE, :NLPBlock, :NLPBlockData, :NLPBoundsPair, :NODE_LIMIT, :NORM_LIMIT, :NUMERICAL_ERROR, :OBJECTIVE_LIMIT, :OPTIMAL, :OPTIMIZE_NOT_CALLED, :OTHER_ERROR, :OTHER_LIMIT, :ObjectiveFunction, :ObjectiveSense, :ObjectiveValue, :OptimizationState, :OptimizationStats, :OptimizerWithAttributes, :RawOptimizerAttribute, :ReInitCache, :ResultCount, :SLOW_PROGRESS, :SOLUTION_LIMIT, :ScalarAffineFunction, :ScalarAffineTerm, :ScalarQuadraticFunction, :ScalarQuadraticTerm, :Silent, :SolveTimeSec, :TIME_LIMIT, :TerminationStatus, :TerminationStatusCode, :TimeLimitSec, :Tunable, :UniversalFallback, :Utilities, :VariableIndex, :VariablePrimal, :VariablePrimalStart, :ZeroOne, :__init, :__solve, :_check_and_convert_maxiters, :_check_and_convert_maxtime, :_process_verbose_param, :add_constraint, :add_variables, :allowscallback, :canonicalize, :constraint_expr, :empty!, :eval_constraint, :eval_constraint_jacobian, :eval_constraint_jacobian_product, :eval_constraint_jacobian_transpose_product, :eval_hessian_lagrangian, :eval_objective, :eval_objective_gradient, :features_available, :get, :get_observed, :get_p, :get_paramsyms, :get_syms, :hessian_lagrangian_structure, :initialize, :instantiate, :instantiate_function, :is_empty, :jacobian_structure, :objective_expr, :optimize!, :requiresconshess, :requiresconsjac, :requiresgradient, :requireshessian, :set, :supports, :supports_incremental_interface, :supports_opt_cache_interface, :supports_sense)),
        all_explicit_imports_are_public = (; ignore = (:varmap_to_vars,)),
    ),
    api_docs_kwargs = (;
        rendered = true,
        docs_src = OPTIMIZATION_DOCS_SRC,
        rendered_ignore = optimization_dependency_rendered_ignore(OptimizationMOI),
        ignore = (
            :AutoModelingToolkit,
            :AutoSparseFastDifferentiation,
            :AutoSparseFiniteDiff,
            :AutoSparseForwardDiff,
            :AutoSparsePolyesterForwardDiff,
            :AutoSparseReverseDiff,
            :AutoSparseZygote,
            :DEFAULT_CALLBACK,
            :DEFAULT_DATA,
            :IncompatibleOptimizerError,
            :MaxSense,
            :MinSense,
            :ObjSense,
            :OptimizationCache,
            :OptimizerMissingError,
        ),
    ),
    ei_broken = (:no_implicit_imports,),
)
