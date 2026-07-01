using OptimizationOptimJL, Aqua, JET
using Test
using Optim

@testset "Aqua" begin
    # OptimizationOptimJL implements the SciML optimization interface for Optim,
    # so the trait/interface methods it adds extend SciML's *own* functions rather
    # than committing type piracy — mark those functions as own.
    SB = OptimizationOptimJL.SciMLBase
    OB = OptimizationOptimJL.OptimizationBase
    Aqua.test_all(
        OptimizationOptimJL;
        piracies = (
            treat_as_own = [
                SB.__init,
                SB.__solve,
                SB.allowsbounds,
                SB.allowscallback,
                SB.allowsconstraints,
                SB.allowsfg,
                SB.has_init,
                SB.requiresbounds,
                SB.requiresconshess,
                SB.requiresconsjac,
                SB.requiresgradient,
                SB.requireshessian,
                OB.supports_sense,
            ],
        )
    )
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationOptimJL; target_defined_modules = true)
end
