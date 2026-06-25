using OptimizationMOI, Aqua, JET
using Test

@testset "Aqua" begin
    # OptimizationMOI deliberately extends SciMLBase's solver-trait interface onto the
    # MathOptInterface optimizer types, so those methods are intentional, not piracy.
    Aqua.test_all(
        OptimizationMOI;
        piracies = (
            treat_as_own = [
                OptimizationMOI.MOI.AbstractOptimizer,
                OptimizationMOI.MOI.OptimizerWithAttributes,
            ],
        )
    )
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationMOI; target_defined_modules = true)
end
