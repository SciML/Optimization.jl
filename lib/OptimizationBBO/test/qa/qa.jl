using SciMLTesting, OptimizationBBO, JET
using Test

# ExplicitImports findings, all tracked against SciML/Optimization.jl:
#  * no_implicit_imports broken: the module relies on `@reexport`/`using`
#    module names (SciMLBase/OptimizationBase/Reexport/...) that cannot be made
#    explicit without restructuring.
#  * the ignored *_are_public / *_via_owners names are owned by SciMLBase,
#    OptimizationBase, the backend, or Base and are not (yet) declared public;
#    the proper fix is upstream `public` declarations, not a local change.
run_qa(
    OptimizationBBO;
    explicit_imports = true,
    ei_kwargs = (;
        all_qualified_accesses_via_owners = (; ignore = (:OptimizationStats,)),
        all_qualified_accesses_are_public = (; ignore = (Symbol("@logmsg"), :LogLevel, :OptRunController, :OptimizationState, :OptimizationStats, :SingleObjectiveMethodNames, :__solve, :_check_and_convert_maxiters, :_check_and_convert_maxtime, :allowsbounds, :allowscallback, :deduce_retcode, :elapsed_time, :num_steps, :requiresbounds, :shutdown_optimizer!)),
    ),
    ei_broken = (:no_implicit_imports,),
)
