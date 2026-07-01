using OptimizationMadNLP, Aqua, JET
using Test

@testset "Aqua" begin
    # `@reexport using OptimizationBase` brings in SciMLBase's `solve!` while
    # `using MadNLP` also pulls a `solve!` into scope; the clash leaves
    # OptimizationMadNLP's `solve!` export pointing at neither binding.
    # Mark broken until the reexport is restructured.
    Aqua.test_all(
        OptimizationMadNLP;
        undefined_exports = (; broken = true)
    )
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationMadNLP; target_defined_modules = true)
end
