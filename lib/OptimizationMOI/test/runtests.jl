using OptimizationMOI, Optimization, Ipopt, NLopt, Zygote, ModelingToolkit
using AmplNLWriter, Ipopt_jll, Juniper
using Test

function _test_sparse_derivatives_hs071(backend, optimizer)
    function objective(x, ::Any)
        return x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]
    end
    function constraints(res, x, ::Any)
        res .= [
            x[1] * x[2] * x[3] * x[4],
            x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2,
        ]
    end
    prob = OptimizationProblem(OptimizationFunction(objective, backend; cons = constraints),
                               [1.0, 5.0, 5.0, 1.0];
                               sense = Optimization.MinSense,
                               lb = [1.0, 1.0, 1.0, 1.0],
                               ub = [5.0, 5.0, 5.0, 5.0],
                               lcons = [25.0, 40.0],
                               ucons = [Inf, 40.0])
    sol = solve(prob, optimizer)
    @test isapprox(sol.objective, 17.014017145179164; atol = 1e-6)
    x = [1.0, 4.7429996418092970, 3.8211499817883077, 1.3794082897556983]
    @test isapprox(sol.minimizer, x; atol = 1e-6)
    @test prod(sol.minimizer) >= 25.0 - 1e-6
    @test isapprox(sum(sol.minimizer .^ 2), 40.0; atol = 1e-6)
    return
end

@testset "OptimizationMOI.jl" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 100.0]
    l1 = rosenbrock(x0, _p)

    optprob = OptimizationFunction((x, p) -> -rosenbrock(x, p), Optimization.AutoZygote())
    prob = OptimizationProblem(optprob, x0, _p; sense = Optimization.MaxSense)

    sol = solve(prob, Ipopt.Optimizer())
    @test 10 * sol.objective < l1

    optprob = OptimizationFunction(rosenbrock, Optimization.AutoZygote())
    prob = OptimizationProblem(optprob, x0, _p; sense = Optimization.MinSense)

    opt = Ipopt.Optimizer()
    sol = solve(prob, opt)
    @test 10 * sol.objective < l1
    sol = solve(prob, opt) #test reuse of optimizer
    @test 10 * sol.objective < l1

    sol = solve(prob,
                OptimizationMOI.MOI.OptimizerWithAttributes(Ipopt.Optimizer,
                                                            "max_cpu_time" => 60.0))
    @test 10 * sol.objective < l1

    sol = solve(prob,
                OptimizationMOI.MOI.OptimizerWithAttributes(NLopt.Optimizer,
                                                            "algorithm" => :LN_BOBYQA))
    @test 10 * sol.objective < l1

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
    optprob = OptimizationFunction(rosenbrock, Optimization.AutoModelingToolkit(true, true);
                                   cons = cons_circ)
    prob = OptimizationProblem(optprob, x0, _p, ucons = [Inf], lcons = [0.0])

    sol = solve(prob, Ipopt.Optimizer())
    @test 10 * sol.objective < l1

    sol = solve(prob,
                OptimizationMOI.MOI.OptimizerWithAttributes(Ipopt.Optimizer,
                                                            "max_cpu_time" => 60.0))
    @test 10 * sol.objective < l1
end

@testset "backends" begin for backend in (Optimization.AutoModelingToolkit(false, false),
                                          Optimization.AutoModelingToolkit(true, false),
                                          Optimization.AutoModelingToolkit(false, true),
                                          Optimization.AutoModelingToolkit(true, true))
    @testset "$backend" begin
        _test_sparse_derivatives_hs071(backend, Ipopt.Optimizer())
        _test_sparse_derivatives_hs071(backend, AmplNLWriter.Optimizer(Ipopt_jll.amplexe))
    end
end end

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
                                      Optimization.AutoForwardDiff())

        optprob = OptimizationProblem(optfun, u0; lb = zero.(u0), ub = one.(u0),
                                      int = ones(Bool, length(u0)),
                                      lcons = [-Inf;], ucons = [W;])

        res = solve(optprob, minlp_solver)
        @test res.u == [0.0, 0.0, 1.0, 0.0]
        @test res.objective == -4.0
    end

    @testset "Integer Domain" begin 
        x = [1.0, 2.0, 4.0, 3.0]
        y = [5.0, 10.0, 20.0, 15.0]
        u0 = [1.0]

        optfun = OptimizationFunction((u, p) -> sum(abs2, x*u[1] .- y), 
                                      Optimization.AutoForwardDiff())

        optprob = OptimizationProblem(optfun, u0; lb = one.(u0), ub = 5.0 .* u0,
                                      int = ones(Bool, length(u0)))

        res = solve(optprob, minlp_solver)
        @test res.u == [4.0]
        @test res.objective ≈ 0.0
    end
end
