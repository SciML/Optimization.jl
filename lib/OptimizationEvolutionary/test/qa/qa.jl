using OptimizationEvolutionary, Aqua, JET
using Test
using Evolutionary

@testset "Aqua" begin
    # SciML trait/interface methods are our own, not piracy — mark them as such.
    # The Evolutionary.trace! override IS genuine piracy (changes Evolutionary's
    # tracing globally); mark the piracy test broken until it's replaced.
    SB = OptimizationEvolutionary.SciMLBase
    Aqua.test_all(
        OptimizationEvolutionary;
        piracies = (
            broken = true,
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
            ],
        )
    )
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationEvolutionary; target_defined_modules = true)
end
