using OptimizationMOI, OptimizationBase, Ipopt, NLopt, Zygote, ModelingToolkit, ReverseDiff
using AmplNLWriter, Ipopt_jll, Juniper, HiGHS, MathOptInterface
using Test, SparseArrays

import MathOptInterface

function _test_sparse_derivatives_hs071(backend, optimizer)
    function objective(x, ::Any)
        return x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]
    end
    function constraints(res, x, ::Any)
        res .= [
            x[1] * x[2] * x[3] * x[4],
            x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2
        ]
    end
    prob = OptimizationProblem(
        OptimizationFunction(objective, backend; cons = constraints),
        [1.0, 5.0, 5.0, 1.0];
        sense = OptimizationBase.MinSense,
        lb = [1.0, 1.0, 1.0, 1.0],
        ub = [5.0, 5.0, 5.0, 5.0],
        lcons = [25.0, 40.0],
        ucons = [Inf, 40.0])
    sol = solve(prob, optimizer)
    @test isapprox(sol.objective, 17.014017145179164; rtol = 1e-1)
    x = [1.0, 4.7429996418092970, 3.8211499817883077, 1.3794082897556983]
    @test isapprox(sol.u, x; rtol = 1e-1)
    @test prod(sol.u) >= 25.0 - 1e-6
    @test isapprox(sum(sol.u .^ 2), 40.0; rtol = 1e-1)
    return
end

@testset "Evaluator" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 100.0]
    cons_circ = (res, x, p) -> res .= [x[1]^2 + x[2]^2]
    optprob = OptimizationFunction(
        rosenbrock, OptimizationBase.AutoZygote();
        cons = cons_circ)
    prob = OptimizationProblem(optprob, x0, _p, ucons = [Inf], lcons = [0.0])
    evaluator = init(prob, Ipopt.Optimizer()).evaluator

    x = prob.u0
    # vector-constraint jacobian product
    @test (evaluator.f.cons_j !== nothing) || (evaluator.f.cons_jvp !== nothing)
    y = zeros(1)
    w = ones(2)
    @test MathOptInterface.eval_constraint_jacobian_product(evaluator, y, x, w) === nothing

    # constraint jacobian-vector product
    @test (evaluator.f.cons_j !== nothing) || (evaluator.f.cons_vjp !== nothing)
    y = zeros(2)
    w = ones(1)
    @test MathOptInterface.eval_constraint_jacobian_transpose_product(
        evaluator, y, x, w) === nothing
end

@testset "NLP" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 100.0]
    l1 = rosenbrock(x0, _p)

    optprob = OptimizationFunction((x, p) -> -rosenbrock(x, p), OptimizationBase.AutoZygote())
    prob = OptimizationProblem(optprob, x0, _p; sense = OptimizationBase.MaxSense)

    callback = function (state, l)
        display(l)
        return false
    end

    sol = solve(prob, Ipopt.Optimizer(); callback)
    @test 10 * sol.objective < l1

    # cache interface
    cache = init(prob, Ipopt.Optimizer())
    sol = solve!(cache)
    @test 10 * sol.objective < l1

    optprob = OptimizationFunction(rosenbrock, OptimizationBase.AutoZygote())
    prob = OptimizationProblem(optprob, x0, _p; sense = OptimizationBase.MinSense)

    opt = Ipopt.Optimizer()
    sol = solve(prob, opt)
    @test 10 * sol.objective < l1
    sol = solve(prob, opt) #test reuse of optimizer
    @test 10 * sol.objective < l1

    # test stats
    @test sol.stats.time > 0
    @test sol.stats.iterations > 0

    sol = solve(prob,
        OptimizationMOI.MOI.OptimizerWithAttributes(Ipopt.Optimizer,
            "max_cpu_time" => 60.0))
    @test 10 * sol.objective < l1

    # test stats with AbstractBridgeOptimizer
    sol = solve(prob,
        OptimizationMOI.MOI.OptimizerWithAttributes(Ipopt.Optimizer,
            "max_cpu_time" => 60.0, "max_iter" => 5))

    @test 60 > sol.stats.time > 0
    @test sol.stats.iterations == 5

    sol = solve(prob,
        OptimizationMOI.MOI.OptimizerWithAttributes(NLopt.Optimizer,
            "algorithm" => :LN_BOBYQA))
    @test 10 * sol.objective < l1

    @test sol.stats.time > 0

    sol = solve(prob,
        OptimizationMOI.MOI.OptimizerWithAttributes(NLopt.Optimizer,
            "algorithm" => :LD_LBFGS))
    @test 10 * sol.objective < l1

    opt = OptimizationMOI.MOI.OptimizerWithAttributes(NLopt.Optimizer,
        "algorithm" => :LD_LBFGS)
    sol = solve(prob, opt)
    @test 10 * sol.objective < l1
    sol = solve(prob, opt)
    @test 10 * sol.objective < l1

    cons_circ = (res, x, p) -> res .= [x[1]^2 + x[2]^2]
    optprob = OptimizationFunction(
        rosenbrock, OptimizationBase.AutoModelingToolkit(true, true);
        cons = cons_circ)
    prob = OptimizationProblem(optprob, x0, _p, ucons = [Inf], lcons = [0.0])

    sol = solve(prob, Ipopt.Optimizer())
    @test 10 * sol.objective < l1

    sol = solve(prob,
        OptimizationMOI.MOI.OptimizerWithAttributes(Ipopt.Optimizer,
            "max_cpu_time" => 60.0))
    @test 10 * sol.objective < l1
end

@testset "backends" begin
    backends = (
        OptimizationBase.AutoModelingToolkit(false, false),
        OptimizationBase.AutoModelingToolkit(true, false),
        OptimizationBase.AutoModelingToolkit(false, true),
        OptimizationBase.AutoModelingToolkit(true, true))
    for backend in backends
        @testset "$backend" begin
            _test_sparse_derivatives_hs071(backend, Ipopt.Optimizer())
            _test_sparse_derivatives_hs071(backend,
                AmplNLWriter.Optimizer(Ipopt_jll.amplexe))
        end
    end
end

@testset "Integer Support" begin
    nl_solver = OptimizationMOI.MOI.OptimizerWithAttributes(Ipopt.Optimizer,
        "print_level" => 0)
    minlp_solver = OptimizationMOI.MOI.OptimizerWithAttributes(Juniper.Optimizer,
        "nl_solver" => nl_solver)

    @testset "Binary Domain" begin
        v = [1.0, 2.0, 4.0, 3.0]
        w = [5.0, 4.0, 3.0, 2.0]
        W = 4.0
        u0 = [0.0, 0.0, 0.0, 1.0]

        optfun = OptimizationFunction((u, p) -> -v'u, cons = (res, u, p) -> res .= w'u,
            OptimizationBase.AutoForwardDiff())

        optprob = OptimizationProblem(optfun, u0; lb = zero.(u0), ub = one.(u0),
            int = ones(Bool, length(u0)),
            lcons = [-Inf;], ucons = [W;])

        res = solve(optprob, minlp_solver)
        @test res.u == [0.0, 0.0, 1.0, 0.0]
        @test res.objective == -4.0
        @test res.stats.time > 0
    end

    @testset "Integer Domain" begin
        x = [1.0, 2.0, 4.0, 3.0]
        y = [5.0, 10.0, 20.0, 15.0]
        u0 = [1.0]

        optfun = OptimizationFunction((u, p) -> sum(abs2, x * u[1] .- y),
            OptimizationBase.AutoForwardDiff())

        optprob = OptimizationProblem(optfun, u0; lb = one.(u0), ub = 6.0 .* u0,
            int = ones(Bool, length(u0)))

        res = solve(optprob, minlp_solver)
        @test res.u ≈ [5.0]
        @test res.objective <= 5eps()
    end
end

@testset "cache" begin
    @variables x
    @parameters a = 1.0
    @named sys = OptimizationSystem((x - a)^2, [x], [a];)
    sys = complete(sys)
    prob = OptimizationProblem(sys, [x => 0.0], []; grad = true, hess = true)
    cache = init(prob, Ipopt.Optimizer(); print_level = 0)
    @test cache isa OptimizationMOI.MOIOptimizationNLPCache
    sol = solve!(cache)
    @test sol.u ≈ [1.0] # ≈ [1]

    @test_broken begin # needs reinit/remake fixes
        cache = OptimizationMOI.reinit!(cache; p = [2.0])
        sol = solve!(cache)
        @test sol.u ≈ [2.0]  # ≈ [2]
    end

    prob = OptimizationProblem(sys, [x => 0.0], []; grad = false, hess = false)
    cache = init(prob, HiGHS.Optimizer())
    @test cache isa OptimizationMOI.MOIOptimizationCache
    sol = solve!(cache)
    @test sol.u≈[1.0] rtol=1e-3 # ≈ [1]

    @test_broken begin
        cache = OptimizationMOI.reinit!(cache; p = [2.0])
        sol = solve!(cache)
        @test sol.u≈[2.0] rtol=1e-3 # ≈ [2]
    end
end

@testset "MOI" begin
    @parameters c = 0.0
    @variables x[1:2]=[0.0, 0.0] [bounds = (c, Inf)]
    @parameters a = 3.0
    @parameters b = 4.0
    @parameters d = 2.0
    @named sys = OptimizationSystem(
        a * x[1]^2 + b * x[2]^2 + d * x[1] * x[2] + 5 * x[1] + x[2], [x...], [a, b, c, d];
        constraints = [
            x[1] + 2 * x[2] ~ 1.0
        ])
    sys = complete(sys)
    prob = OptimizationProblem(sys, [x[1] => 2.0, x[2] => 0.0], []; grad = true,
        hess = true)
    sol = solve(prob, HiGHS.Optimizer())
    sol.u

    @named sys = OptimizationSystem(
    a * x[1]^2 + b * x[2]^2 + d * x[1] * x[2] + 5 * x[1] + x[2], [x...], [a, b, c, d];
    constraints = [
        x[1] + 2 * x[2] ~ 1.0
        x[1] ≲ 1
        -1.0 ≲ x[2]
    ])
    sys = complete(sys)
    prob = OptimizationProblem(sys, [x[1] => 2.0, x[2] => 0.0], []; grad = true,
        hess = true)
    sol = solve(prob, HiGHS.Optimizer())
    sol.u
end

@testset "tutorial" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 1.0]

    cons(res, x, p) = (res .= [x[1]^2 + x[2]^2, x[1] * x[2]])

    optprob = OptimizationFunction(rosenbrock, OptimizationBase.AutoModelingToolkit();
        cons = cons)
    prob = OptimizationProblem(optprob, x0, _p, lcons = [1.0, 0.5], ucons = [1.0, 0.5])
    sol = solve(prob, AmplNLWriter.Optimizer(Ipopt_jll.amplexe))
end

@testset "tutorial" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 1.0]

    cons(res, x, p) = (res .= [x[1]^2 + x[2]^2, x[1] * x[2]])

    function lagh(res, x, sigma, mu, p)
        lH = sigma * [2 + 8(x[1]^2) * p[2]-4(x[2] - (x[1]^2)) * p[2] -4p[2]*x[1]
              -4p[2]*x[1] 2p[2]] .+ [2mu[1] mu[2]
              mu[2] 2mu[1]]
        res .= lH[[1, 3, 4]]
    end
    lag_hess_prototype = sparse([1 1; 0 1])

    optprob = OptimizationFunction(rosenbrock, OptimizationBase.AutoForwardDiff();
        cons = cons, lag_h = lagh, lag_hess_prototype)
    prob = OptimizationProblem(optprob, x0, _p, lcons = [1.0, 0.5], ucons = [1.0, 0.5])
    sol = solve(prob, Ipopt.Optimizer())
end

@testset "common maxiters interface" begin
    # Test that the common maxiters interface works without warnings
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 100.0]
    
    optprob = OptimizationFunction(rosenbrock, Optimization.AutoZygote())
    prob = OptimizationProblem(optprob, x0, _p)
    
    # Test with Ipopt using maxiters parameter
    @testset "Ipopt maxiters" begin
        # This should not produce a warning and should respect the iteration limit
        sol = solve(prob, Ipopt.Optimizer(); maxiters = 5, print_level = 0)
        # Should terminate due to iteration limit
        @test sol.stats.iterations <= 5
    end
    
    # Test with cache interface
    @testset "Cache interface maxiters" begin
        cache = init(prob, Ipopt.Optimizer(); maxiters = 3, print_level = 0)
        sol = solve!(cache)
        @test sol.stats.iterations <= 3
    end
    
    # Test that unknown solver fallback works gracefully
    @testset "Generic fallback" begin
        # Mock optimizer that doesn't match any known pattern
        struct MockOptimizer <: MathOptInterface.AbstractOptimizer end
        
        # This should not error, but may show a warning for unknown solver
        mock_opt = MockOptimizer()
        # We can't actually solve with this mock optimizer, but we can test
        # that the parameter setting doesn't crash
        @test_nowarn OptimizationMOI._set_maxiters!(mock_opt, 10)
    end
end
