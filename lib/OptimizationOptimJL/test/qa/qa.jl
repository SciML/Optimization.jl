using OptimizationOptimJL, Aqua, JET
using Test
using Optim

@testset "Aqua" begin
    # Extending SciMLBase traits onto Optim optimizer types is the entire purpose
    # of this package, so flag those types as "our own" for Aqua's piracy check.
    Aqua.test_all(
        OptimizationOptimJL;
        piracies = (
            treat_as_own = [
                Optim.AbstractOptimizer,
                Optim.ConstrainedOptimizer,
                Optim.Fminbox,
                Optim.IPNewton,
                Optim.KrylovTrustRegion,
                Optim.Newton,
                Optim.NewtonTrustRegion,
                Optim.SAMIN,
                Optim.SimulatedAnnealing,
            ],
        )
    )
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationOptimJL; target_defined_modules = true)
end
