using SciMLTesting, OptimizationMetaheuristics, JET
using Test
using Metaheuristics

# ExplicitImports findings, all tracked against SciML/Optimization.jl:
#  * no_implicit_imports broken: the module relies on `@reexport`/`using`
#    module names (SciMLBase/OptimizationBase/Reexport/...) that cannot be made
#    explicit without restructuring.
#  * the ignored *_are_public / *_via_owners names are owned by SciMLBase,
#    OptimizationBase, the backend, or Base and are not (yet) declared public;
#    the proper fix is upstream `public` declarations, not a local change.
# OptimizationMetaheuristics implements the SciML optimization interface for
# Metaheuristics, so the trait/interface methods it adds extend SciML's *own*
# functions rather than committing type piracy — mark those functions as own.
# undefined_exports broken (tracked against SciML/Optimization.jl):
# `@reexport using Metaheuristics` exports `solve!`, which clashes with SciMLBase's
# `solve!` brought in transitively via OptimizationBase; mark broken until restructured.
SB = OptimizationMetaheuristics.SciMLBase
include(normpath(joinpath(@__DIR__, "..", "..", "..", "..", "test", "qa", "public_api_docs.jl")))

run_qa(
    OptimizationMetaheuristics;
    explicit_imports = true,
    aqua_kwargs = (;
        piracies = (;
            treat_as_own = [
                SB.__init,
                SB.__solve,
                SB.allowsbounds,
                SB.allowscallback,
                SB.has_init,
                SB.requiresbounds,
            ],
        ),
    ),
    aqua_broken = (:undefined_exports,),
    ei_kwargs = (;
        all_qualified_accesses_via_owners = (; ignore = (:OptimizationStats,)),
        all_qualified_accesses_are_public = (; ignore = (:AbstractAlgorithm, :OptimizationStats, :__init, :__solve, :_check_and_convert_maxiters, :_check_and_convert_maxtime, :allowscallback, :create_child, :get_best, :requiresbounds)),
    ),
    api_docs_kwargs = public_api_docs_kwargs(OptimizationMetaheuristics),
    ei_broken = (:no_implicit_imports,),
)
