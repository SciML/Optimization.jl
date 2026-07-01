using OptimizationMetaheuristics, Aqua, JET
using Test
using Metaheuristics

@testset "Aqua" begin
    # Extending SciMLBase traits onto Metaheuristics optimizer types is the entire purpose
    # of this package, so flag those types as "our own" for Aqua's piracy check.
    # `@reexport using Metaheuristics` exports `solve!`, which clashes with SciMLBase's
    # `solve!` brought in transitively via OptimizationBase; mark broken until restructured.
    Aqua.test_all(
        OptimizationMetaheuristics;
        piracies = (
            treat_as_own = [
                Metaheuristics.AbstractAlgorithm,
            ],
        ),
        undefined_exports = (; broken = true)
    )
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationMetaheuristics; target_defined_modules = true)
end
