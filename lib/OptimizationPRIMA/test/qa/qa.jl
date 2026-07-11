using SciMLTesting, OptimizationPRIMA, JET
using Test

# ExplicitImports findings, all tracked against SciML/Optimization.jl:
#  * no_implicit_imports broken: the module relies on `@reexport`/`using`
#    module names (SciMLBase/OptimizationBase/Reexport/...) that cannot be made
#    explicit without restructuring.
#  * the ignored *_are_public / *_via_owners names are owned by SciMLBase,
#    OptimizationBase, the backend, or Base and are not (yet) declared public;
#    the proper fix is upstream `public` declarations, not a local change.
include(normpath(joinpath(@__DIR__, "..", "..", "..", "..", "test", "qa", "public_api_docs.jl")))

run_qa(
    OptimizationPRIMA;
    explicit_imports = true,
    ei_kwargs = (;
        all_qualified_accesses_via_owners = (; ignore = (:OptimizationStats,)),
        all_qualified_accesses_are_public = (; ignore = (:AnalysisResults, :DAMAGING_ROUNDING, :FTARGET_ACHIEVED, :INVALID_INPUT, :MAXFUN_REACHED, :MAXTR_REACHED, :NAN_INF_F, :NAN_INF_MODEL, :NAN_INF_X, :NO_SPACE_BETWEEN_BOUNDS, :NoAD, :OptimizationState, :OptimizationStats, :ReInitCache, :SMALL_TR_RADIUS, :Status, :ZERO_LINEAR_CONSTRAINT, :__solve, :_check_and_convert_maxiters, :_check_and_convert_maxtime, :_process_verbose_param, :allowscallback, :apply_sense, :instantiate_function, :requiresconshess, :requiresconsjac, :requiresconstraints, :supports_sense)),
    ),
    api_docs_kwargs = public_api_docs_kwargs(OptimizationPRIMA),
    ei_broken = (:no_implicit_imports,),
)
