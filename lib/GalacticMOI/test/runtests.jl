using GalacticMOI, GalacticOptim, Ipopt, NLopt, Zygote, ModelingToolkit
using Test

@testset "GalacticMOI.jl" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 100.0]
    l1 = rosenbrock(x0, _p)

    optprob = OptimizationFunction((x, p) -> -rosenbrock(x, p), GalacticOptim.AutoZygote())
    prob = OptimizationProblem(optprob, x0, _p; sense=GalacticOptim.MaxSense)

    sol = solve(prob, Ipopt.Optimizer())
    @test 10 * sol.minimum < l1

    optprob = OptimizationFunction(rosenbrock, GalacticOptim.AutoZygote())
    prob = OptimizationProblem(optprob, x0, _p; sense=GalacticOptim.MinSense)

    sol = solve(prob, Ipopt.Optimizer())
    @test 10 * sol.minimum < l1

    sol = solve(prob, GalacticMOI.MOI.OptimizerWithAttributes(Ipopt.Optimizer, "max_cpu_time" => 60.0))
    @test 10 * sol.minimum < l1

    sol = solve(prob, GalacticMOI.MOI.OptimizerWithAttributes(NLopt.Optimizer, "algorithm" => :LN_BOBYQA))
    @test 10 * sol.minimum < l1

    sol = solve(prob, GalacticMOI.MOI.OptimizerWithAttributes(NLopt.Optimizer, "algorithm" => :LD_LBFGS))
    @test 10 * sol.minimum < l1

    sol = solve(prob, GalacticMOI.MOI.OptimizerWithAttributes(NLopt.Optimizer, "algorithm" => :LD_LBFGS))
    @test 10 * sol.minimum < l1

    cons_circ = (x, p) -> [x[1]^2 + x[2]^2]
    optprob = OptimizationFunction(rosenbrock, GalacticOptim.AutoModelingToolkit(true, true); cons=cons_circ)
    prob = OptimizationProblem(optprob, x0, _p)

    sol = solve(prob, Ipopt.Optimizer())
    @test 10 * sol.minimum < l1

    sol = solve(prob, GalacticMOI.MOI.OptimizerWithAttributes(Ipopt.Optimizer, "max_cpu_time" => 60.0))
    @test 10 * sol.minimum < l1
end
