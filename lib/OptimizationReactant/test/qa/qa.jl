using SciMLTesting, OptimizationReactant, JET, SciMLBase
using Test

include(normpath(joinpath(@__DIR__, "..", "..", "..", "..", "test", "qa", "rendered_docs.jl")))

using OptimizationBase

# OptimizationReactant implements the `instantiate_function` extension point of
# OptimizationBase, so those methods extend OptimizationBase's own function —
# mark it as own for the Aqua piracy check. The Reactant names it uses
# (`to_rarray`, `compile`, the `AbstractConcrete*` supertypes) are documented
# API but not yet declared `public` upstream.
run_qa(
    OptimizationReactant;
    explicit_imports = true,
    aqua_kwargs = (;
        piracies = (;
            treat_as_own = [OptimizationBase.instantiate_function],
        ),
    ),
    ei_kwargs = (;
        all_explicit_imports_are_public = (;
            ignore = (:ReInitCache, :instantiate_function),
        ),
        all_qualified_accesses_are_public = (;
            ignore = (
                :AbstractConcreteArray, :AbstractConcreteNumber,
                :compile, :to_number, :to_rarray,
            ),
        ),
    ),
    reexports_allow = (:AutoReactant,),
)
