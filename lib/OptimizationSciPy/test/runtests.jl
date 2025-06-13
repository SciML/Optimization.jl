using OptimizationSciPy, Optimization, Zygote, ReverseDiff, ForwardDiff
using Test, Random
using Optimization.SciMLBase: ReturnCode
using PythonCall


function rosenbrock(x, p)
    (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
end

function rosenbrock_hess(H, x, p)
    H[1,1] = 2 - 400*p[2]*x[2] + 1200*p[2]*x[1]^2
    H[1,2] = -400*p[2]*x[1]
    H[2,1] = -400*p[2]*x[1]
    H[2,2] = 200*p[2]
    return nothing
end

@testset "OptimizationSciPy.jl" begin
    x0 = zeros(2)
    _p = [1.0, 100.0]
    l1 = rosenbrock(x0, _p)

    
    optprob = OptimizationFunction((x, p) -> -rosenbrock(x, p), Optimization.AutoZygote())
    prob = OptimizationProblem(optprob, x0, _p; sense = Optimization.MaxSense)
    sol = solve(prob, ScipyNelderMead())
    @test sol.retcode == ReturnCode.Success
    @test 10 * sol.objective < l1

    
    optprob = OptimizationFunction(rosenbrock, Optimization.AutoZygote())
    prob = OptimizationProblem(optprob, x0, _p)

    sol = solve(prob, ScipyBFGS())
    @test sol.retcode == ReturnCode.Success
    @test 10 * sol.objective < l1

    sol = solve(prob, ScipyLBFGSB())
    @test sol.retcode == ReturnCode.Success
    @test 10 * sol.objective < l1

    
    prob = OptimizationProblem(optprob, x0, _p, lb = [-1.0, -1.0], ub = [0.8, 0.8])

    sol = solve(prob, ScipyLBFGSB())
    @test sol.retcode == ReturnCode.Success
    @test 10 * sol.objective < l1

    
    sol = solve(prob, ScipyDifferentialEvolution(), maxiters = 100)
    @test sol.retcode == ReturnCode.Success
    @test 10 * sol.objective < l1

    sol = solve(prob, ScipyBasinhopping(), maxiters = 50)
    @test sol.retcode == ReturnCode.Success
    @test 10 * sol.objective < l1

    sol = solve(prob, ScipyDualAnnealing(), maxiters = 100)
    @test sol.retcode == ReturnCode.Success
    @test 10 * sol.objective < l1

    sol = solve(prob, ScipyShgo())
    @test sol.retcode == ReturnCode.Success
    @test 10 * sol.objective < l1

    sol = solve(prob, ScipyDirect(), maxiters = 1000)
    @test sol.retcode in (ReturnCode.Success, ReturnCode.Failure)
    if sol.retcode == ReturnCode.Success
        @test 10 * sol.objective < l1
    end

    sol = solve(prob, ScipyBrute(), Ns = 10)
    @test sol.retcode == ReturnCode.Success
    @test 10 * sol.objective < l1

    
    optprob = OptimizationFunction(rosenbrock, Optimization.AutoZygote())
    prob = OptimizationProblem(optprob, x0, _p)
    
    sol = solve(prob, ScipyNelderMead())
    @test sol.retcode == ReturnCode.Success
    @test 10 * sol.objective < l1

    sol = solve(prob, ScipyPowell())
    @test sol.retcode == ReturnCode.Success
    @test 10 * sol.objective < l1

    sol = solve(prob, ScipyCG())
    @test sol.retcode == ReturnCode.Success
    @test 10 * sol.objective < l1

    sol = solve(prob, ScipyTNC())
    @test sol.retcode == ReturnCode.Success
    @test 10 * sol.objective < l1

    
    optf = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff(); hess = rosenbrock_hess)
    prob = OptimizationProblem(optf, x0, _p)
    sol = solve(prob, ScipyNewtonCG(), maxiters = 200)
    @test sol.retcode == ReturnCode.Success
    @test 10 * sol.objective < l1

    
    optprob = OptimizationFunction(rosenbrock, Optimization.AutoZygote())
    prob = OptimizationProblem(optprob, x0, _p, lb = [-1.0, -1.0], ub = [0.8, 0.8])
    
    sol = solve(prob, ScipyLBFGSB())
    @test sol.retcode == ReturnCode.Success
    @test 10 * sol.objective < l1

    sol = solve(prob, ScipyTNC())
    @test sol.retcode == ReturnCode.Success
    @test 10 * sol.objective < l1

    
    optf_hess = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff(); hess = rosenbrock_hess)
    prob = OptimizationProblem(optf_hess, x0, _p)
    
    for method in [ScipyDogleg(), ScipyTrustNCG(), ScipyTrustKrylov(), ScipyTrustExact()]
        sol = solve(prob, method, maxiters = 200)
        @test sol.retcode == ReturnCode.Success
        @test 10 * sol.objective < l1
    end

    @testset "cache interface" begin
        objective(x, p) = (p[1] - x[1])^2
        x0 = zeros(1)
        p = [1.0]

        optf = OptimizationFunction(objective, Optimization.AutoZygote())
        prob = OptimizationProblem(optf, x0, p)
        cache = Optimization.init(prob, ScipyBFGS())
        sol = Optimization.solve!(cache)
        @test sol.retcode == ReturnCode.Success
        @test sol.u ≈ [1.0] atol=1e-3

        cache = Optimization.reinit!(cache; p = [2.0])
        sol = Optimization.solve!(cache)
        @test sol.u ≈ [2.0] atol=1e-3
    end

    @testset "callback" begin
        cbstopping = function (state, loss)
            return state.objective < 0.7
        end

        optprob = OptimizationFunction(rosenbrock, Optimization.AutoZygote())
        prob = OptimizationProblem(optprob, x0, _p)
        
        @test_throws ErrorException solve(prob, ScipyBFGS(), callback = cbstopping)
    end

    @testset "constrained optimization" begin
        Random.seed!(1)
        cons = (res, x, p) -> res .= [x[1]^2 + x[2]^2 - 1.0]
        cons_j = (res, x, p) -> begin
            res[1,1] = 2*x[1]
            res[1,2] = 2*x[2]
        end
        x0 = zeros(2)
        optprob = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff(); cons = cons, cons_j = cons_j)
        
        
        prob_cobyla = OptimizationProblem(optprob, x0, _p, lcons = [-1e-6], ucons = [1e-6])
        sol = solve(prob_cobyla, ScipyCOBYLA(), maxiters = 10000)
        @test sol.retcode == ReturnCode.Success
        @test 10 * sol.objective < l1

        
        Random.seed!(42)
        prob = OptimizationProblem(optprob, rand(2), _p, lcons = [0.0], ucons = [0.0])
        sol = solve(prob, ScipySLSQP())
        @test sol.retcode == ReturnCode.Success
        @test 10 * sol.objective < l1

        Random.seed!(123)
        prob = OptimizationProblem(optprob, rand(2), _p, lcons = [0.0], ucons = [0.0])
        sol = solve(prob, ScipyTrustConstr())
        @test sol.retcode == ReturnCode.Success
        @test 10 * sol.objective < l1

        
        function con2_c(res, x, p)
            res .= [x[1]^2 + x[2]^2 - 1.0, x[2] * sin(x[1]) - x[1] - 2.0]
        end

        optprob = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff(); cons = con2_c)
        Random.seed!(456)
        prob = OptimizationProblem(optprob, rand(2), _p, lcons = [0.0, -Inf], ucons = [0.0, 0.0])
        
        sol = solve(prob, ScipySLSQP())
        @test sol.retcode == ReturnCode.Success
        @test 10 * sol.objective < l1

        
        Random.seed!(789)
        prob = OptimizationProblem(optprob, [0.5, 0.5], _p, lcons = [-Inf, -Inf],
            ucons = [0.0, 0.0], lb = [-1.0, -1.0], ub = [1.0, 1.0])
        sol = solve(prob, ScipyShgo(), n = 50, iters = 1)
        @test sol.retcode == ReturnCode.Success
        @test sol.objective < l1
    end

    @testset "method-specific options" begin
        simple_optprob = OptimizationFunction(rosenbrock, Optimization.AutoZygote())
        unconstrained_prob = OptimizationProblem(simple_optprob, x0, _p, lb = [-1.0, -1.0], ub = [1.0, 1.0])
        
        
        sol = solve(unconstrained_prob, ScipyDifferentialEvolution(), 
                   popsize = 10, mutation = (0.5, 1.0), recombination = 0.7)
        @test sol.retcode == ReturnCode.Success
        
        
        sol = solve(unconstrained_prob, ScipyBasinhopping(), T = 1.0, stepsize = 0.5, niter = 10)
        @test sol.retcode == ReturnCode.Success
        
        
        sol = solve(unconstrained_prob, ScipyDualAnnealing(), 
                   initial_temp = 5000.0, restart_temp_ratio = 2e-5)
        @test sol.retcode == ReturnCode.Success
        
        
        sol = solve(unconstrained_prob, ScipyShgo(), n = 50, sampling_method = "simplicial")
        @test sol.retcode == ReturnCode.Success
        
        
        sol = solve(unconstrained_prob, ScipyDirect(), eps = 0.001, locally_biased = true)
        @test sol.retcode == ReturnCode.Success
        
        
        sol = solve(unconstrained_prob, ScipyBrute(), Ns = 5, workers = 1)
        @test sol.retcode == ReturnCode.Success
    end

    @testset "gradient-free methods" begin
        
        optf_no_grad = OptimizationFunction(rosenbrock)
        prob = OptimizationProblem(optf_no_grad, x0, _p)
        
        
        sol = solve(prob, ScipyCOBYLA(), maxiters = 10000)
        @test sol.retcode == ReturnCode.Success
        
        
        sol = solve(prob, ScipyNelderMead())
        @test sol.retcode == ReturnCode.Success
        
        
        sol = solve(prob, ScipyPowell())
        @test sol.retcode == ReturnCode.Success
    end

    @testset "AutoDiff backends" begin
        
        for adtype in [Optimization.AutoZygote(), 
                      Optimization.AutoReverseDiff(), 
                      Optimization.AutoForwardDiff()]
            optf = OptimizationFunction(rosenbrock, adtype)
            prob = OptimizationProblem(optf, x0, _p)
            sol = solve(prob, ScipyBFGS())
            @test sol.retcode == ReturnCode.Success
            @test 10 * sol.objective < l1
        end
    end

    @testset "optimization stats" begin
        optprob = OptimizationFunction(rosenbrock, Optimization.AutoZygote())
        prob = OptimizationProblem(optprob, x0, _p)
        sol = solve(prob, ScipyBFGS())
        @test sol.stats.time > 0
    end

    @testset "original result access" begin
        optprob = OptimizationFunction(rosenbrock, Optimization.AutoZygote())
        prob = OptimizationProblem(optprob, x0, _p)
        sol = solve(prob, ScipyBFGS())
        @test !isnothing(sol.original)
        @test pyhasattr(sol.original, "success")
        @test pyhasattr(sol.original, "message")
    end

    @testset "tolerance settings" begin
        optprob = OptimizationFunction(rosenbrock, Optimization.AutoZygote())
        prob = OptimizationProblem(optprob, x0, _p)
        
        
        sol = solve(prob, ScipyNelderMead(), abstol = 1e-8)
        @test sol.objective < 1e-7
        
        
        sol = solve(prob, ScipyBFGS(), reltol = 1e-8)
        @test sol.objective < 1e-7
    end

    @testset "constraint satisfaction" begin
        cons = (res, x, p) -> res .= [x[1]^2 + x[2]^2 - 1.0]
        optprob = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff(); cons = cons)
        prob = OptimizationProblem(optprob, [0.5, 0.5], _p, lcons = [-0.01], ucons = [0.01])
        
        sol = solve(prob, ScipySLSQP())
        @test sol.retcode == ReturnCode.Success
        
        
        cons_val = [0.0]
        cons(cons_val, sol.u, _p)
        @test abs(cons_val[1]) < 0.011
    end

    @testset "invalid method" begin
        @test_throws ArgumentError ScipyMinimize("InvalidMethodName")
    end

    @testset "Edge cases" begin
        
        f_simple(x, p) = (x[1] - p[1])^2
        prob = OptimizationProblem(f_simple, [0.0], [3.0])
        sol = solve(prob, ScipyBFGS())
        @test sol.retcode == ReturnCode.Success
        @test sol.u ≈ [3.0] atol=1e-6
        
        
        optprob = OptimizationFunction(rosenbrock, Optimization.AutoZygote())
        prob = OptimizationProblem(optprob, x0, _p)
        @test_throws SciMLBase.IncompatibleOptimizerError solve(prob, ScipyDifferentialEvolution())
        @test_throws SciMLBase.IncompatibleOptimizerError solve(prob, ScipyDirect())
        @test_throws SciMLBase.IncompatibleOptimizerError solve(prob, ScipyDualAnnealing())
        @test_throws SciMLBase.IncompatibleOptimizerError solve(prob, ScipyBrute())
    end
end