using OptimizationMultistartOptimization, Aqua, JET
using Test
using MultistartOptimization

@testset "Aqua" begin
    # Extending SciMLBase traits onto MultistartOptimization optimizer types is the
    # entire purpose of this package, so flag those types as "our own" for Aqua's
    # piracy check.
    Aqua.test_all(
        OptimizationMultistartOptimization;
        piracies = (
            treat_as_own = [
                MultistartOptimization.TikTak,
            ],
        ),
        # OptimizationNLopt is used in tests as the inner solver, not in src.
        stale_deps = (; ignore = [:OptimizationNLopt])
    )
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationMultistartOptimization; target_defined_modules = true)
end
