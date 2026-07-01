using OptimizationNLopt, Aqua, JET
using Test
using NLopt

@testset "Aqua" begin
    # OptimizationNLopt implements the SciML optimization interface for NLopt, so
    # the trait/interface methods it adds extend SciML's *own* functions rather
    # than committing type piracy — mark those functions as own for Aqua's piracy
    # check. NLopt.Algorithm is also kept for the `(::NLopt.Algorithm)()`
    # normalization method, which extends NLopt's type directly and has no SciML
    # function to attribute it to.
    SB = OptimizationNLopt.SciMLBase
    OB = OptimizationNLopt.OptimizationBase
    Aqua.test_all(
        OptimizationNLopt;
        piracies = (
            treat_as_own = [
                SB.__init,
                SB.__solve,
                SB.allowsbounds,
                SB.allowscallback,
                SB.allowsconstraints,
                SB.has_init,
                SB.requiresconsjac,
                SB.requiresgradient,
                SB.requireshessian,
                OB.supports_sense,
                NLopt.Algorithm,
            ],
        )
    )
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationNLopt; target_defined_modules = true)
end
