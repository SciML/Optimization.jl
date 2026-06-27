using SciMLTesting, OptimizationNLPModels, JET
using Test

# ExplicitImports findings, all tracked against SciML/Optimization.jl:
#  * no_implicit_imports broken: the module relies on `@reexport`/`using`
#    module names (SciMLBase/OptimizationBase/Reexport/...) that cannot be made
#    explicit without restructuring.
#  * the ignored *_are_public / *_via_owners names are owned by SciMLBase,
#    OptimizationBase, the backend, or Base and are not (yet) declared public;
#    the proper fix is upstream `public` declarations, not a local change.
run_qa(
    OptimizationNLPModels;
    explicit_imports = true,
    aqua_kwargs = (;
        # The sublibrary extends SciMLBase's solver-trait/__init/__solve interface
        # onto its backend's optimizer types, so those methods are intentional.
        piracies = (;
            treat_as_own = [
                OptimizationNLPModels.ADTypes.AbstractADType,
                OptimizationNLPModels.NLPModels.AbstractNLPModel,
                OptimizationNLPModels.SciMLBase.OptimizationFunction,
                OptimizationNLPModels.SciMLBase.OptimizationProblem,
            ],
        ),
    ),
    ei_kwargs = (;
        # `NoAD` is owned by SciMLBase and remains non-public there on the registered
        # release (SciMLBase 3.24.0); fix belongs upstream via a `public` declaration.
        all_qualified_accesses_are_public = (; ignore = (:NoAD,)),
    ),
    ei_broken = (:no_implicit_imports,),
)
