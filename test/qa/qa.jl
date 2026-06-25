using SciMLTesting, Optimization, JET
using Test

# no_implicit_imports: Optimization is a facade that `@reexport`s SciMLBase/ADTypes/
# OptimizationBase and `using`s many helper modules; the implicit-import surface is
# large and intentional. Tracked as known-broken (SciML/Optimization.jl).
# no_stale_explicit_imports ignores: `instantiate_function` is reached as
# `Optimization.instantiate_function` by downstream/tests; `ReInitCache` and
# `OptimizationStats` are part of the intentionally re-surfaced cache/stats API.
# *_are_public ignores: these names are owned by SciMLBase/OptimizationBase, which
# have not (yet) declared them public, so the non-public flag is on the source pkg.
run_qa(
    Optimization;
    explicit_imports = true,
    aqua_kwargs = (;
        ambiguities = (; recursive = false),
        piracies = (;
            treat_as_own = [
                Optimization.SciMLBase.OptimizationProblem,
                Optimization.SciMLBase.AbstractOptimizationCache,
            ],
        ),
    ),
    ei_kwargs = (;
        no_stale_explicit_imports = (;
            ignore = (:instantiate_function, :ReInitCache, :OptimizationStats),
        ),
        all_explicit_imports_are_public = (;
            ignore = (
                :MaxSense, :MinSense, :ObjSense, :OptimizationStats,
                :ReInitCache, :instantiate_function,
            ),
        ),
    ),
    ei_broken = (:no_implicit_imports,),
)
