using OptimizationCMAEvolutionStrategy, OptimizationBase
using OptimizationBase: SciMLBase
using Test

@testset "OptimizationCMAEvolutionStrategy.jl" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 100.0]
    l1 = rosenbrock(x0, _p)
    f = OptimizationFunction(rosenbrock)
    prob = OptimizationProblem(f, x0, _p, lb = [-1.0, -1.0], ub = [0.8, 0.8])
    sol = solve(prob, CMAEvolutionStrategyOpt())
    @test 10 * sol.objective < l1
    @test sol.retcode isa SciMLBase.ReturnCode.T

    function cb(state, args...)
        if state.iter % 10 == 0
            println(state.u)
        end
        return false
    end
    sol = solve(prob, CMAEvolutionStrategyOpt(), callback = cb, maxiters = 100)
    @test sol.u == OptimizationCMAEvolutionStrategy.CMAEvolutionStrategy.xbest(sol.original)
    @test sol.retcode isa SciMLBase.ReturnCode.T

    # Force `:maxiter` with a very small iteration budget so CMA can't converge.
    sol_maxit = solve(prob, CMAEvolutionStrategyOpt(), maxiters = 2)
    @test sol_maxit.retcode == SciMLBase.ReturnCode.MaxIters

    @testset "_cma_retcode symbol mapping" begin
        _cma_retcode = OptimizationCMAEvolutionStrategy._cma_retcode
        @test _cma_retcode(:ftarget) == SciMLBase.ReturnCode.Success
        @test _cma_retcode(:xtol) == SciMLBase.ReturnCode.Success
        @test _cma_retcode(:ftol) == SciMLBase.ReturnCode.Success
        @test _cma_retcode(:maxiter) == SciMLBase.ReturnCode.MaxIters
        @test _cma_retcode(:maxfevals) == SciMLBase.ReturnCode.MaxIters
        @test _cma_retcode(:maxtime) == SciMLBase.ReturnCode.MaxTime
        @test _cma_retcode(:stagnation) == SciMLBase.ReturnCode.Stalled
        @test _cma_retcode(:none) == SciMLBase.ReturnCode.Default
    end
end
