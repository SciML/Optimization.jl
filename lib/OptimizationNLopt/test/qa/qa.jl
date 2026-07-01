using OptimizationNLopt, Aqua, JET
using Test
using NLopt

@testset "Aqua" begin
    # Extending SciMLBase traits onto NLopt optimizer types is the entire purpose
    # of this package, so flag those types as "our own" for Aqua's piracy check.
    Aqua.test_all(
        OptimizationNLopt;
        piracies = (
            treat_as_own = [
                NLopt.Algorithm,
                NLopt.Opt,
            ],
        )
    )
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationNLopt; target_defined_modules = true)
end
