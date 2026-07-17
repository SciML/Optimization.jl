using SciMLTesting, Optimization, JET, SciMLBase
using Test

include("rendered_docs.jl")

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
                SciMLBase.OptimizationProblem,
                SciMLBase.AbstractOptimizationCache,
            ],
        ),
    ),
    ei_kwargs = (;
        no_stale_explicit_imports = (;
            ignore = (:instantiate_function, :ReInitCache, :OptimizationStats),
        ),
        all_explicit_imports_are_public = (;
            # Owned by SciMLBase (MaxSense/MinSense/ObjSense/OptimizationStats) and
            # OptimizationBase (ReInitCache/instantiate_function); not yet declared
            # public in their source pkgs (verified non-public on SciMLBase 3.24.0 /
            # OptimizationBase 5.1.3). Fix belongs upstream via `public` declarations.
            ignore = (
                :MaxSense, :MinSense, :ObjSense, :OptimizationStats,
                :ReInitCache, :instantiate_function,
            ),
        ),
    ),
    api_docs_kwargs = (;
        rendered = true,
        docs_src = OPTIMIZATION_DOCS_SRC,
        rendered_ignore = optimization_dependency_rendered_ignore(Optimization),
        ignore = (
            :AutoModelingToolkit,
            :AutoSparseFastDifferentiation,
            :AutoSparseFiniteDiff,
            :AutoSparseForwardDiff,
            :AutoSparsePolyesterForwardDiff,
            :AutoSparseReverseDiff,
            :AutoSparseZygote,
        ),
    ),
    ei_broken = (:no_implicit_imports,),
)
