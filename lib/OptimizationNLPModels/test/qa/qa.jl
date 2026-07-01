using OptimizationNLPModels, Aqua, JET
using Test
using NLPModels

@testset "Aqua" begin
    # Defining SciMLBase.OptimizationFunction / SciMLBase.OptimizationProblem constructors
    # for NLPModels.AbstractNLPModel is the entire purpose of this package, so flag that
    # type as "our own" for Aqua's piracy check.
    Aqua.test_all(
        OptimizationNLPModels;
        piracies = (
            treat_as_own = [NLPModels.AbstractNLPModel],
        ),
        # Optimization sublibraries used to construct OptimizationProblems in tests
        # but not `using`d in src; Aqua's static analysis can't see test-only usage.
        stale_deps = (; ignore = [:OptimizationLBFGSB, :OptimizationMOI, :OptimizationOptimJL])
    )
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationNLPModels; target_defined_modules = true)
end
