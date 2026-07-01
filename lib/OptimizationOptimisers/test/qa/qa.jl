using OptimizationOptimisers, Aqua, JET
using Test
using Optimisers

@testset "Aqua" begin
    # Extending SciMLBase traits onto Optimisers optimizer types is the entire purpose
    # of this package, so flag those types as "our own" for Aqua's piracy check.
    Aqua.test_all(
        OptimizationOptimisers;
        piracies = (
            treat_as_own = [
                Optimisers.AbstractRule,
            ],
        )
    )
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationOptimisers; target_defined_modules = true)
end
