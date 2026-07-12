using SciMLTesting, OptimizationOptimisers, JET, SciMLBase
using Test
using Optimisers

# ExplicitImports findings, all tracked against SciML/Optimization.jl:
#  * no_implicit_imports broken: the module relies on `@reexport`/`using`
#    module names (SciMLBase/OptimizationBase/Reexport/...) that cannot be made
#    explicit without restructuring.
#  * the ignored *_are_public / *_via_owners names are owned by SciMLBase,
#    OptimizationBase, the backend, or Base and are not (yet) declared public;
#    the proper fix is upstream `public` declarations, not a local change.
# OptimizationOptimisers implements the SciML optimization interface for
# Optimisers, so the trait/interface methods it adds extend SciML's *own*
# functions rather than committing type piracy — mark those functions as own.
run_qa(
    OptimizationOptimisers;
    explicit_imports = true,
    aqua_kwargs = (;
        piracies = (;
            treat_as_own = [
                SciMLBase.__init,
                SciMLBase.__solve,
                SciMLBase.allowscallback,
                SciMLBase.allowsfg,
                SciMLBase.has_init,
                SciMLBase.requiresgradient,
            ],
        ),
    ),
    ei_kwargs = (;
        all_qualified_accesses_via_owners = (; ignore = (:OptimizationStats,)),
        all_qualified_accesses_are_public = (; ignore = (:OptimizationState, :OptimizationStats, :__init, :__solve, :_check_and_convert_maxiters, :allowscallback, :allowsfg, :isa_dataiterator, :requiresgradient)),
    ),
    api_docs_kwargs = (;
        ignore = (
            :ADADelta,
            :ADAGrad,
            :ADAM,
            :ADAMW,
            :AbstractRule,
            :AutoModelingToolkit,
            :AutoSparseFastDifferentiation,
            :AutoSparseFiniteDiff,
            :AutoSparseForwardDiff,
            :AutoSparsePolyesterForwardDiff,
            :AutoSparseReverseDiff,
            :AutoSparseZygote,
            :NADAM,
            :OADAM,
            :RADAM,
        ),
    ),
    ei_broken = (:no_implicit_imports,),
)
