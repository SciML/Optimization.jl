using SciMLTesting, OptimizationNLopt, JET
using Test
using NLopt

# ExplicitImports findings, all tracked against SciML/Optimization.jl:
#  * no_implicit_imports broken: the module relies on `@reexport`/`using`
#    module names (SciMLBase/OptimizationBase/Reexport/...) that cannot be made
#    explicit without restructuring.
#  * the ignored *_are_public / *_via_owners names are owned by SciMLBase,
#    OptimizationBase, the backend, or Base and are not (yet) declared public;
#    the proper fix is upstream `public` declarations, not a local change.
# OptimizationNLopt implements the SciML optimization interface for NLopt, so
# the trait/interface methods it adds extend SciML's *own* functions rather
# than committing type piracy — mark those functions as own for Aqua's piracy
# check. NLopt.Algorithm is also kept for the `(::NLopt.Algorithm)()`
# normalization method, which extends NLopt's type directly and has no SciML
# function to attribute it to.
SB = OptimizationNLopt.SciMLBase
OB = OptimizationNLopt.OptimizationBase
include(normpath(joinpath(@__DIR__, "..", "..", "..", "..", "test", "qa", "public_api_docs.jl")))

run_qa(
    OptimizationNLopt;
    explicit_imports = true,
    aqua_kwargs = (;
        piracies = (;
            treat_as_own = [
                SB.__init,
                SB.__solve,
                SB.allowsbounds,
                SB.allowscallback,
                SB.allowsconstraints,
                SB.has_init,
                SB.requiresconsjac,
                SB.requiresgradient,
                SB.requireshessian,
                OB.supports_sense,
                NLopt.Algorithm,
            ],
        ),
    ),
    ei_kwargs = (;
        all_qualified_accesses_via_owners = (; ignore = (:OptimizationStats,)),
        all_qualified_accesses_are_public = (; ignore = (:AUGLAG, :LD_AUGLAG, :LN_AUGLAG, :OptimizationState, :OptimizationStats, :__init, :__solve, :_check_and_convert_maxiters, :_check_and_convert_maxtime, :allowscallback, :nlopt_set_param, :requiresconsjac, :requiresgradient, :requireshessian, :supports_sense)),
        all_explicit_imports_are_public = (; ignore = (:deduce_retcode,)),
    ),
    api_docs_kwargs = public_api_docs_kwargs(OptimizationNLopt),
    ei_broken = (:no_implicit_imports,),
)
