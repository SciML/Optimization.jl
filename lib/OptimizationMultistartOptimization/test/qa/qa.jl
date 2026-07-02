using OptimizationMultistartOptimization, Aqua, JET
using Test
using MultistartOptimization

@testset "Aqua" begin
    # OptimizationMultistartOptimization implements the SciML optimization interface
    # for MultistartOptimization, so the trait/interface methods it adds extend
    # SciML's *own* functions rather than committing type piracy — mark those
    # functions as own.
    SB = OptimizationMultistartOptimization.SciMLBase
    Aqua.test_all(
        OptimizationMultistartOptimization;
        piracies = (
            treat_as_own = [
                SB.__init,
                SB.__solve,
                SB.allowsbounds,
                SB.allowscallback,
                SB.has_init,
                SB.requiresbounds,
            ],
        ),
        # OptimizationNLopt is used in tests as the inner solver, not in src.
        stale_deps = (; ignore = [:OptimizationNLopt])
    )
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationMultistartOptimization; target_defined_modules = true)
end
