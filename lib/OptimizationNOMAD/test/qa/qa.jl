using OptimizationNOMAD, Aqua, JET
using Test

@testset "Aqua" begin
    # `@reexport using OptimizationBase` exports `solve!`, which clashes with NOMAD's
    # `solve!` brought in via `using NOMAD`; mark broken until the reexport is restructured.
    Aqua.test_all(
        OptimizationNOMAD;
        undefined_exports = (; broken = true)
    )
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationNOMAD; target_defined_modules = true)
end
