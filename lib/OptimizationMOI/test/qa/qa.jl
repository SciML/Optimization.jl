using OptimizationMOI, Aqua, JET
using Test
using MathOptInterface

@testset "Aqua" begin
    # OptimizationMOI implements the SciML optimization interface for
    # MathOptInterface, so the trait/interface methods it adds extend SciML's *own*
    # functions rather than committing type piracy — mark those functions as own.
    SB = OptimizationMOI.SciMLBase
    OB = OptimizationMOI.OptimizationBase
    Aqua.test_all(
        OptimizationMOI;
        piracies = (
            treat_as_own = [
                SB.__init,
                SB.__solve,
                SB.allowsbounds,
                SB.allowscallback,
                SB.allowsconstraints,
                SB.get_observed,
                SB.get_p,
                SB.get_paramsyms,
                SB.get_syms,
                SB.has_init,
                SB.requiresconshess,
                SB.requiresconsjac,
                SB.requiresgradient,
                SB.requireshessian,
                SB.supports_opt_cache_interface,
                OB.supports_sense,
            ],
        )
    )
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationMOI; target_defined_modules = true)
end
