using OptimizationMetaheuristics, Aqua, JET
using Test
using Metaheuristics

@testset "Aqua" begin
    # OptimizationMetaheuristics implements the SciML optimization interface for
    # Metaheuristics, so the trait/interface methods it adds extend SciML's *own*
    # functions rather than committing type piracy — mark those functions as own.
    # `@reexport using Metaheuristics` exports `solve!`, which clashes with SciMLBase's
    # `solve!` brought in transitively via OptimizationBase; mark broken until restructured.
    SB = OptimizationMetaheuristics.SciMLBase
    Aqua.test_all(
        OptimizationMetaheuristics;
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
        undefined_exports = (; broken = true)
    )
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationMetaheuristics; target_defined_modules = true)
end
