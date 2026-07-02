using OptimizationOptimisers, Aqua, JET
using Test
using Optimisers

@testset "Aqua" begin
    # OptimizationOptimisers implements the SciML optimization interface for
    # Optimisers, so the trait/interface methods it adds extend SciML's *own*
    # functions rather than committing type piracy — mark those functions as own.
    SB = OptimizationOptimisers.SciMLBase
    Aqua.test_all(
        OptimizationOptimisers;
        piracies = (
            treat_as_own = [
                SB.__init,
                SB.__solve,
                SB.allowscallback,
                SB.allowsfg,
                SB.has_init,
                SB.requiresgradient,
            ],
        )
    )
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationOptimisers; target_defined_modules = true)
end
