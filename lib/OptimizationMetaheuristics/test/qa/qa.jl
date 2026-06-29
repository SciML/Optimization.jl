using SciMLTesting, OptimizationMetaheuristics, JET
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
# Conflicting export: `summary` (Metaheuristics + Base).
run_qa(
    OptimizationMetaheuristics;
    explicit_imports = true,
    aqua_kwargs = (;
        # The sublibrary extends SciMLBase's solver-trait/__init/__solve interface
        # onto its backend's optimizer types, so those methods are intentional.
        piracies = (;
            treat_as_own = [
                OptimizationMetaheuristics.Metaheuristics.AbstractAlgorithm,
                OptimizationMetaheuristics.OptimizationCache,
                OptimizationMetaheuristics.SciMLBase.OptimizationProblem,
            ],
        ),
    ),
    aqua_broken = (:undefined_exports,),
    # jet_broken (tracked against SciML/Optimization.jl): JET typo-mode flags
    # `local variable `opt_bounds` may be undefined` in `__solve`
    # (src/OptimizationMetaheuristics.jl) — `opt_bounds` is assigned only inside the
    # `!isnothing(cache.lb) & !isnothing(cache.ub)` guard but read unconditionally,
    # so an unbounded problem would be undefined. Pre-existing latent issue surfaced
    # by enabling JET; the solver-correctness fix is handled separately.
    jet_broken = true,
    ei_kwargs = (;
        all_qualified_accesses_via_owners = (; ignore = (:OptimizationStats,)),
        all_qualified_accesses_are_public = (; ignore = (:AbstractAlgorithm, :OptimizationStats, :__init, :__solve, :_check_and_convert_maxiters, :_check_and_convert_maxtime, :allowscallback, :create_child, :get_best, :requiresbounds)),
    ),
    ei_broken = (:no_implicit_imports,),
)
