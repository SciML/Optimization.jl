using SciMLTesting, OptimizationNOMAD, JET
using Test

# ExplicitImports findings, all tracked against SciML/Optimization.jl:
#  * no_implicit_imports broken: the module relies on `@reexport`/`using`
#    module names (SciMLBase/OptimizationBase/Reexport/...) that cannot be made
#    explicit without restructuring.
#  * the ignored *_are_public / *_via_owners names are owned by SciMLBase,
#    OptimizationBase, the backend, or Base and are not (yet) declared public;
#    the proper fix is upstream `public` declarations, not a local change.
# undefined_exports broken (tracked against SciML/Optimization.jl): the module
# `@reexport`s two packages that both export the name below, leaving it an
# unresolved conflict binding; the fix is a maintainer choice of which wins.
# Conflicting export: `solve` (NOMAD + CommonSolve).
# jet_broken (tracked against SciML/Optimization.jl): JET typo-mode flags
# `local variable `bb`/`bbcons` may be undefined` in `__solve`
# (src/OptimizationNOMAD.jl) — the black-box closures are each defined in only one
# arm of the `prob.f.cons === nothing` if/else and consumed in the matching arm, so
# the two conditionals are correlated but JET cannot prove it. Pre-existing latent
# issue surfaced by enabling JET; resolving the smell is a separate src refactor.
run_qa(
    OptimizationNOMAD;
    explicit_imports = true,
    aqua_broken = (:undefined_exports,),
    jet_broken = true,
    ei_kwargs = (;
        all_qualified_accesses_via_owners = (; ignore = (:OptimizationStats,)),
        all_qualified_accesses_are_public = (; ignore = (:DefaultOptimizationCache, :OptimizationStats, :__solve, :_check_and_convert_maxiters, :_check_and_convert_maxtime, :allowscallback)),
    ),
    ei_broken = (:no_implicit_imports,),
)
