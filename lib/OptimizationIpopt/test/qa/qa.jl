using SciMLTesting, OptimizationIpopt, JET
using Test

# ExplicitImports findings, all tracked against SciML/Optimization.jl:
#  * no_implicit_imports broken: the module relies on `@reexport`/`using`
#    module names (SciMLBase/OptimizationBase/Reexport/...) that cannot be made
#    explicit without restructuring.
#  * the ignored *_are_public / *_via_owners names are owned by SciMLBase,
#    OptimizationBase, the backend, or Base and are not (yet) declared public;
#    the proper fix is upstream `public` declarations, not a local change.
run_qa(
    OptimizationIpopt;
    explicit_imports = true,
    ei_kwargs = (;
        all_qualified_accesses_via_owners = (; ignore = (:OptimizationStats,)),
        all_qualified_accesses_are_public = (; ignore = (Symbol("@logmsg"), :ApplicationReturnStatus, :Diverging_Iterates, :Feasible_Point_Found, :GetIpoptCurrentIterate, :Infeasible_Problem_Detected, :LogLevel, :Maximum_CpuTime_Exceeded, :Maximum_Iterations_Exceeded, :Maximum_WallTime_Exceeded, :OptimizationState, :OptimizationStats, :Search_Direction_Becomes_Too_Small, :Solve_Succeeded, :Solved_To_Acceptable_Level, :User_Requested_Stop, :__solve, :_check_and_convert_maxiters, :_check_and_convert_maxtime, :allowscallback, :instantiate_function, :requiresconshess, :requiresconsjac, :requiresgradient, :requireshessian, :requireslagh, :supports_sense)),
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
        ),
    ),
    ei_broken = (:no_implicit_imports,),
)
