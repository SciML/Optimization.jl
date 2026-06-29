using SciMLTesting, OptimizationNLopt, JET
using Test

# ExplicitImports findings, all tracked against SciML/Optimization.jl:
#  * no_implicit_imports broken: the module relies on `@reexport`/`using`
#    module names (SciMLBase/OptimizationBase/Reexport/...) that cannot be made
#    explicit without restructuring.
#  * the ignored *_are_public / *_via_owners names are owned by SciMLBase,
#    OptimizationBase, the backend, or Base and are not (yet) declared public;
#    the proper fix is upstream `public` declarations, not a local change.
# jet_broken (tracked against SciML/Optimization.jl): JET typo-mode flags
# `local variable `thetacache`/`Jthetacache` is not defined` inside the
# constraint-callback closures of `__solve` (src/OptimizationNLopt.jl). Both are
# assigned before the closures (lines 199-200) but reassigned *inside* them, so
# Julia boxes the captured binding and JET's closure analysis cannot prove the box
# is initialized — a benign closure-capture artifact, not a runtime bug.
# Pre-existing latent issue surfaced by enabling JET; resolving it is a separate
# closure refactor, not part of the QA conversion.
run_qa(
    OptimizationNLopt;
    explicit_imports = true,
    jet_broken = true,
    aqua_kwargs = (;
        # The sublibrary extends SciMLBase's solver-trait/__init/__solve interface
        # onto its backend's optimizer types, so those methods are intentional.
        piracies = (;
            treat_as_own = [
                OptimizationNLopt.NLopt.Algorithm,
                OptimizationNLopt.NLopt.Opt,
                OptimizationNLopt.OptimizationCache,
                OptimizationNLopt.SciMLBase.OptimizationProblem,
            ],
        ),
    ),
    ei_kwargs = (;
        all_qualified_accesses_via_owners = (; ignore = (:OptimizationStats,)),
        all_qualified_accesses_are_public = (; ignore = (:AUGLAG, :LD_AUGLAG, :LN_AUGLAG, :OptimizationState, :OptimizationStats, :__init, :__solve, :_check_and_convert_maxiters, :_check_and_convert_maxtime, :allowscallback, :nlopt_set_param, :requiresconsjac, :requiresgradient, :requireshessian, :supports_sense)),
        all_explicit_imports_are_public = (; ignore = (:deduce_retcode,)),
    ),
    ei_broken = (:no_implicit_imports,),
)
