using SciMLTesting, OptimizationNLopt, JET, OptimizationBase, SciMLBase
using Test

include(normpath(joinpath(@__DIR__, "..", "..", "..", "..", "test", "qa", "rendered_docs.jl")))

using NLopt

# Kept in sync with the reexport `export` block in src/OptimizationNLopt.jl. NLopt's
# algorithm constants (`NLopt.LD_LBFGS` and friends) are not exported by NLopt itself,
# so the exported `NLopt` module binding is what makes the documented spelling work.
const NLOPT_REEXPORTS = (:Algorithm, :NLopt, :Opt)

# ExplicitImports findings, all tracked against SciML/Optimization.jl:
#  * no_implicit_imports broken: the module relies on `using`
#    module names (SciMLBase/OptimizationBase/...) that cannot be made
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
run_qa(
    OptimizationNLopt;
    explicit_imports = true,
    aqua_kwargs = (;
        piracies = (;
            treat_as_own = [
                SciMLBase.__init,
                SciMLBase.__solve,
                SciMLBase.allowsbounds,
                SciMLBase.allowscallback,
                SciMLBase.allowsconstraints,
                SciMLBase.has_init,
                SciMLBase.requiresconsjac,
                SciMLBase.requiresgradient,
                SciMLBase.requireshessian,
                OptimizationBase.supports_sense,
                NLopt.Algorithm,
            ],
        ),
    ),
    ei_kwargs = (;
        all_qualified_accesses_via_owners = (; ignore = (:OptimizationStats,)),
        all_qualified_accesses_are_public = (; ignore = (:AUGLAG, :LD_AUGLAG, :LN_AUGLAG, :OptimizationState, :OptimizationStats, :__init, :__solve, :_check_and_convert_maxiters, :_check_and_convert_maxtime, :allowscallback, :nlopt_set_param, :requiresconsjac, :requiresgradient, :requireshessian, :supports_sense)),
        all_explicit_imports_are_public = (; ignore = (:deduce_retcode,)),
    ),
    ei_broken = (:no_implicit_imports,),
    reexports_allow = NLOPT_REEXPORTS,
)

@testset "reexported NLopt names" begin
    test_reexported_names(OptimizationNLopt, NLOPT_REEXPORTS)
end
