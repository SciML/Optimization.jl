using SciMLTesting, OptimizationOptimisers, JET, SciMLBase
using Test

include(normpath(joinpath(@__DIR__, "..", "..", "..", "..", "test", "qa", "rendered_docs.jl")))

using Optimisers

# Kept in sync with the reexport `export` block in src/OptimizationOptimisers.jl: the
# Optimisers rule names `using OptimizationOptimisers` is expected to bring into scope.
const OPTIMISERS_REEXPORTS = (
    :ADADelta, :ADAGrad, :ADAM, :ADAMW, :AMSGrad, :AccumGrad,
    :AdaBelief, :AdaDelta, :AdaGrad, :AdaMax, :Adam, :AdamW, :ClipGrad, :ClipNorm,
    :Descent, :Lion, :Momentum, :NADAM, :NAdam, :Nesterov, :OADAM, :OAdam,
    :OptimiserChain, :Optimisers, :RADAM, :RAdam, :RMSProp, :Rprop, :SignDecay,
    :WeightDecay,
)

# ExplicitImports findings, all tracked against SciML/Optimization.jl:
#  * no_implicit_imports broken: the module relies on `using`
#    module names (SciMLBase/OptimizationBase/...) that cannot be made
#    explicit without restructuring.
#  * the ignored *_are_public / *_via_owners names are owned by SciMLBase,
#    OptimizationBase, the backend, or Base and are not (yet) declared public;
#    the proper fix is upstream `public` declarations, not a local change.
# OptimizationOptimisers implements the SciML optimization interface for
# Optimisers, so the trait/interface methods it adds extend SciML's *own*
# functions rather than committing type piracy — mark those functions as own.
run_qa(
    OptimizationOptimisers;
    explicit_imports = true,
    aqua_kwargs = (;
        piracies = (;
            treat_as_own = [
                SciMLBase.__init,
                SciMLBase.__solve,
                SciMLBase.allowscallback,
                SciMLBase.allowsfg,
                SciMLBase.has_init,
                SciMLBase.requiresgradient,
            ],
        ),
    ),
    ei_kwargs = (;
        all_qualified_accesses_via_owners = (; ignore = (:OptimizationStats,)),
        all_qualified_accesses_are_public = (; ignore = (:OptimizationState, :OptimizationStats, :__init, :__solve, :_check_and_convert_maxiters, :allowscallback, :allowsfg, :isa_dataiterator, :requiresgradient)),
    ),
    ei_broken = (:no_implicit_imports,),
    # Optimisers.jl gives its legacy all-caps aliases no docstring of their own; they
    # are kept public here so code written against `ADAM(0.01)` keeps working. The
    # docstring obligation is Optimisers.jl's.
    api_docs_kwargs = (;
        ignore = (:ADADelta, :ADAGrad, :ADAM, :ADAMW, :NADAM, :OADAM, :RADAM),
    ),
    reexports_allow = OPTIMISERS_REEXPORTS,
)

@testset "reexported Optimisers rule names" begin
    test_reexported_names(OptimizationOptimisers, OPTIMISERS_REEXPORTS)
end
