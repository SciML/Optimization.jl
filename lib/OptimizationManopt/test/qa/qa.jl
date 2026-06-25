using SciMLTesting, OptimizationManopt, JET
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
# Conflicting export: `solve!` (Manopt + OptimizationBase).
run_qa(
    OptimizationManopt;
    explicit_imports = true,
    aqua_broken = (:undefined_exports,),
    ei_kwargs = (;
        all_qualified_accesses_via_owners = (; ignore = (:AbstractManifold,)),
        all_qualified_accesses_are_public = (; ignore = (:AbstractManifold, :Failure, :MaxIters, :MaxTime, :OptimizationState, :Stalled, :Success, :Unstable, :__solve, :allowscallback, :build_solution, :has_init, :requiresgradient, :requireshessian, :supports_sense)),
    ),
    ei_broken = (:no_implicit_imports,),
)
