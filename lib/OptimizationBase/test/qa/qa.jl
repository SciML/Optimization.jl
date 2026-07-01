using OptimizationBase, Aqua, JET
using Test

@testset "Aqua" begin
    # OptimizationBase defines the Optimization solve interface itself, so it
    # extends SciMLBase's __init/__solve and CommonSolve's init/solve/solve! (the
    # latter reexported through SciMLBase) on SciMLBase.OptimizationProblem /
    # AbstractOptimizationCache. Those are our *own* interface functions, so mark
    # them as own for the piracy check rather than flagging the SciML types.
    SB = OptimizationBase.SciMLBase
    Aqua.test_all(
        OptimizationBase;
        piracies = (
            treat_as_own = [SB.__init, SB.__solve, SB.init, SB.solve, SB.solve!],
        )
    )
end

@testset "JET static analysis" begin
    JET.test_package(OptimizationBase; target_defined_modules = true)
end
