using OptimizationMOI, Optimization, Ipopt, NLopt, Zygote, ModelingToolkit
using AmplNLWriter, Ipopt_jll
using Test

function _test_sparse_derivatives_hs071(backend, optimizer)
    function objective(x, ::Any)
        return x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]
    end
    function constraints(x, ::Any)
        return [x[1] * x[2] * x[3] * x[4], x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2]
    end
    prob = OptimizationProblem(
        OptimizationFunction(objective, backend; cons = constraints),
        [1.0, 5.0, 5.0, 1.0];
        sense = Optimization.MinSense,
        lb = [1.0, 1.0, 1.0, 1.0],
        ub = [5.0, 5.0, 5.0, 5.0],
        lcons = [25.0, 40.0],
        ucons = [Inf, 40.0],
    )
    sol = solve(prob, optimizer)
    @test isapprox(sol.minimum, 17.014017145179164; atol = 1e-6)
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
    @test 10 * sol.minimum < l1

    optprob = OptimizationFunction(rosenbrock, Optimization.AutoZygote())
    prob = OptimizationProblem(optprob, x0, _p; sense = Optimization.MinSense)

    sol = solve(prob, Ipopt.Optimizer())
    @test 10 * sol.minimum < l1

    sol = solve(
        prob,
        OptimizationMOI.MOI.OptimizerWithAttributes(
            Ipopt.Optimizer,
            "max_cpu_time" => 60.0,
        ),
    )
    @test 10 * sol.minimum < l1

    sol = solve(
        prob,
        OptimizationMOI.MOI.OptimizerWithAttributes(
            NLopt.Optimizer,
            "algorithm" => :LN_BOBYQA,
        ),
    )
    @test 10 * sol.minimum < l1

    sol = solve(
        prob,
        OptimizationMOI.MOI.OptimizerWithAttributes(
            NLopt.Optimizer,
            "algorithm" => :LD_LBFGS,
        ),
    )
    @test 10 * sol.minimum < l1

    sol = solve(
        prob,
        OptimizationMOI.MOI.OptimizerWithAttributes(
            NLopt.Optimizer,
            "algorithm" => :LD_LBFGS,
        ),
    )
    @test 10 * sol.minimum < l1

    cons_circ = (x, p) -> [x[1]^2 + x[2]^2]
    optprob = OptimizationFunction(
        rosenbrock,
        Optimization.AutoModelingToolkit(true, true);
        cons = cons_circ,
    )
    prob = OptimizationProblem(optprob, x0, _p, ucons = [Inf], lcons = [0.0])

    sol = solve(prob, Ipopt.Optimizer())
    @test 10 * sol.minimum < l1

    sol = solve(
        prob,
        OptimizationMOI.MOI.OptimizerWithAttributes(
            Ipopt.Optimizer,
            "max_cpu_time" => 60.0,
        ),
    )
    @test 10 * sol.minimum < l1
end

@testset "backends" begin
    for backend in (
        Optimization.AutoModelingToolkit(false, false),
        Optimization.AutoModelingToolkit(true, false),
        Optimization.AutoModelingToolkit(false, true),
        Optimization.AutoModelingToolkit(true, true),
    )
        @testset "$backend" begin
            _test_sparse_derivatives_hs071(backend, Ipopt.Optimizer())
            _test_sparse_derivatives_hs071(
                backend,
                AmplNLWriter.Optimizer(Ipopt_jll.amplexe),
            )
        end
    end
end
