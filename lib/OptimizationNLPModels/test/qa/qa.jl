using OptimizationNLPModels, Aqua, JET
using Test
using NLPModels

@testset "Aqua" begin
    # This package defines SciMLBase.OptimizationFunction / SciMLBase.OptimizationProblem
    # constructors for NLPModels.AbstractNLPModel. Those constructors extend SciML's
    # *own* types, so mark those (not the NLPModels type) as own for the piracy check.
    SB = OptimizationNLPModels.SciMLBase
    Aqua.test_all(
        OptimizationNLPModels;
        piracies = (
            treat_as_own = [SB.OptimizationFunction, SB.OptimizationProblem],
        ),
        # Optimization sublibraries used to construct OptimizationProblems in tests
        # but not `using`d in src; Aqua's static analysis can't see test-only usage.
        stale_deps = (; ignore = [:OptimizationLBFGSB, :OptimizationMOI, :OptimizationOptimJL])
    )
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationNLPModels; target_defined_modules = true)
end
