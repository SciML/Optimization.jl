using OptimizationEvolutionary, Aqua, JET
using Test
using Evolutionary

@testset "Aqua" begin
    # Extending SciMLBase traits onto Evolutionary optimizer types is the entire purpose
    # of this package, so flag those types as "our own" for Aqua's piracy check.
    Aqua.test_all(
        OptimizationEvolutionary;
        piracies = (
            treat_as_own = [
                Evolutionary.AbstractOptimizer,
                Evolutionary.NSGA2,
            ],
        )
    )
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationEvolutionary; target_defined_modules = true)
end
