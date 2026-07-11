using SciMLTesting, OptimizationIpopt, JET
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
    OptimizationIpopt;
    explicit_imports = true,
    ei_kwargs = (;
        all_qualified_accesses_via_owners = (; ignore = (:OptimizationStats,)),
        all_qualified_accesses_are_public = (; ignore = (Symbol("@logmsg"), :AbstractOptimizationCache, :ApplicationReturnStatus, :DEFAULT_VERBOSE, :Diverging_Iterates, :Feasible_Point_Found, :GetIpoptCurrentIterate, :Infeasible_Problem_Detected, :LogLevel, :Maximum_CpuTime_Exceeded, :Maximum_Iterations_Exceeded, :Maximum_WallTime_Exceeded, :OptimizationState, :OptimizationStats, :ReInitCache, :Search_Direction_Becomes_Too_Small, :Solve_Succeeded, :Solved_To_Acceptable_Level, :User_Requested_Stop, :__init, :__solve, :_check_and_convert_maxiters, :_check_and_convert_maxtime, :_process_verbose_param, :allowscallback, :get_observed, :get_p, :get_paramsyms, :get_syms, :instantiate_function, :requiresconshess, :requiresconsjac, :requiresgradient, :requireshessian, :supports_opt_cache_interface, :supports_sense)),
    ),
    api_docs_kwargs = public_api_docs_kwargs(OptimizationIpopt),
    ei_broken = (:no_implicit_imports,),
)
