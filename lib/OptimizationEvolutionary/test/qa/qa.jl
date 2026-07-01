using OptimizationEvolutionary, Aqua, JET
using Test
using Evolutionary

@testset "Aqua" begin
    # OptimizationEvolutionary implements the SciML optimization interface for
    # Evolutionary, so the trait/interface methods it adds extend SciML's *own*
    # functions rather than committing type piracy — mark those functions as own.
    # Evolutionary.trace! is also extended (to feed Evolutionary's tracing into
    # our callbacks); it has no SciML function to attribute it to, so mark that
    # function as own too.
    SB = OptimizationEvolutionary.SciMLBase
    Aqua.test_all(
        OptimizationEvolutionary;
        piracies = (
            treat_as_own = [
                SB.__solve,
                SB.allowsbounds,
                SB.allowscallback,
                SB.allowsconstraints,
                SB.has_init,
                SB.requiresconshess,
                SB.requiresconsjac,
                SB.requiresgradient,
                SB.requireshessian,
                Evolutionary.trace!,
            ],
        )
    )
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationEvolutionary; target_defined_modules = true)
end
