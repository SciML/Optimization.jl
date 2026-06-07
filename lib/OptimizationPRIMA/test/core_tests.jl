using OptimizationPRIMA, OptimizationBase, ForwardDiff, ModelingToolkit, ReverseDiff
using OptimizationBase: SciMLBase
using Test

@testset "OptimizationPRIMA.jl" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 100.0]
    l1 = rosenbrock(x0, _p)

    prob = OptimizationProblem(rosenbrock, x0, _p)
    sol = OptimizationBase.solve(prob, UOBYQA(), maxiters = 1000)
    @test 10 * sol.objective < l1
    @test sol.retcode == SciMLBase.ReturnCode.Success
    sol = OptimizationBase.solve(prob, NEWUOA(), maxiters = 1000)
    @test 10 * sol.objective < l1
    @test sol.retcode == SciMLBase.ReturnCode.Success
    sol = OptimizationBase.solve(prob, BOBYQA(), maxiters = 1000)
    @test 10 * sol.objective < l1
    @test sol.retcode == SciMLBase.ReturnCode.Success
    sol = OptimizationBase.solve(prob, LINCOA(), maxiters = 1000)
    @test 10 * sol.objective < l1
    @test sol.retcode == SciMLBase.ReturnCode.Success
    @test_throws OptimizationBase.IncompatibleOptimizerError OptimizationBase.solve(
        prob, COBYLA(), maxiters = 1000
    )

    # Force MaxIters with a very small iteration budget — also exercises a
    # non-FTARGET_ACHIEVED/SMALL_TR_RADIUS return path.
    sol_maxit = OptimizationBase.solve(prob, UOBYQA(), maxiters = 3)
    @test sol_maxit.retcode == SciMLBase.ReturnCode.MaxIters

    @testset "sciml_prima_retcode enum mapping" begin
        m = OptimizationPRIMA.sciml_prima_retcode
        # PRIMA treats both SMALL_TR_RADIUS and FTARGET_ACHIEVED as success.
        @test m(PRIMA.FTARGET_ACHIEVED) == SciMLBase.ReturnCode.Success
        @test m(PRIMA.SMALL_TR_RADIUS) == SciMLBase.ReturnCode.Success
        @test m(PRIMA.MAXFUN_REACHED) == SciMLBase.ReturnCode.MaxIters
        @test m(PRIMA.MAXTR_REACHED) == SciMLBase.ReturnCode.MaxIters
        @test m(PRIMA.NO_SPACE_BETWEEN_BOUNDS) == SciMLBase.ReturnCode.InitialFailure
        @test m(PRIMA.ZERO_LINEAR_CONSTRAINT) == SciMLBase.ReturnCode.InitialFailure
        @test m(PRIMA.INVALID_INPUT) == SciMLBase.ReturnCode.InitialFailure
        @test m(PRIMA.NAN_INF_X) == SciMLBase.ReturnCode.Unstable
        @test m(PRIMA.NAN_INF_F) == SciMLBase.ReturnCode.Unstable
        @test m(PRIMA.NAN_INF_MODEL) == SciMLBase.ReturnCode.Unstable
        @test m(PRIMA.DAMAGING_ROUNDING) == SciMLBase.ReturnCode.ConvergenceFailure
        @test m(PRIMA.TRSUBP_FAILED) == SciMLBase.ReturnCode.Failure
        @test m(PRIMA.ASSERTION_FAILS) == SciMLBase.ReturnCode.Failure
        @test m(PRIMA.VALIDATION_FAILS) == SciMLBase.ReturnCode.Failure
        @test m(PRIMA.MEMORY_ALLOCATION_FAILS) == SciMLBase.ReturnCode.Failure
    end

    function prima_con_2c(res, x, p)
        res .= [x[1] + x[2], x[2] * sin(x[1]) - x[1]]
    end
    optprob = OptimizationFunction(rosenbrock, AutoForwardDiff(), cons = prima_con_2c)
    prob = OptimizationProblem(optprob, x0, _p, lcons = [1, -100], ucons = [1, 100])
    sol = OptimizationBase.solve(prob, COBYLA(), maxiters = 1000)
    @test sol.objective < l1

    function prima_con_1c(res, x, p)
        res .= [x[1] + x[2]]
    end
    optprob = OptimizationFunction(rosenbrock, AutoForwardDiff(), cons = prima_con_1c)
    prob = OptimizationProblem(optprob, x0, _p, lcons = [1], ucons = [1])
    sol = OptimizationBase.solve(prob, COBYLA(), maxiters = 1000)
    @test sol.objective < l1

    prob = OptimizationProblem(optprob, x0, _p, lcons = [1], ucons = [5])
    sol = OptimizationBase.solve(prob, COBYLA(), maxiters = 1000)
    @test sol.objective < l1

    function prima_con_nl(res, x, p)
        res .= [x[2] * sin(x[1]) - x[1]]
    end
    optprob = OptimizationFunction(rosenbrock, AutoSymbolics(), cons = prima_con_nl)
    prob = OptimizationProblem(optprob, x0, _p, lcons = [10], ucons = [50])
    sol = OptimizationBase.solve(prob, COBYLA(), maxiters = 1000)
    @test 10 * sol.objective < l1
end
