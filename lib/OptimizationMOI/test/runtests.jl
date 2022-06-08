using OptimizationMOI, Optimization, Ipopt, NLopt, Zygote, ModelingToolkit, Juniper, AmplNLWriter, Ipopt_jll
using Test

@testset "OptimizationMOI.jl" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 100.0]
    l1 = rosenbrock(x0, _p)

    optprob = OptimizationFunction((x, p) -> -rosenbrock(x, p), Optimization.AutoZygote())
    prob = OptimizationProblem(optprob, x0, _p; sense=Optimization.MaxSense)

    sol = solve(prob, Ipopt.Optimizer())
    @test 10 * sol.minimum < l1

    optprob = OptimizationFunction(rosenbrock, Optimization.AutoZygote())
    prob = OptimizationProblem(optprob, x0, _p; sense=Optimization.MinSense)

    sol = solve(prob, Ipopt.Optimizer())
    @test 10 * sol.minimum < l1

    sol = solve(prob, OptimizationMOI.MOI.OptimizerWithAttributes(Ipopt.Optimizer, "max_cpu_time" => 60.0))
    @test 10 * sol.minimum < l1

    sol = solve(prob, OptimizationMOI.MOI.OptimizerWithAttributes(NLopt.Optimizer, "algorithm" => :LN_BOBYQA))
    @test 10 * sol.minimum < l1

    sol = solve(prob, OptimizationMOI.MOI.OptimizerWithAttributes(NLopt.Optimizer, "algorithm" => :LD_LBFGS))
    @test 10 * sol.minimum < l1

    sol = solve(prob, OptimizationMOI.MOI.OptimizerWithAttributes(NLopt.Optimizer, "algorithm" => :LD_LBFGS))
    @test 10 * sol.minimum < l1

    cons_circ = (x, p) -> [x[1]^2 + x[2]^2]
    optprob = OptimizationFunction(rosenbrock, Optimization.AutoModelingToolkit(true, true); cons=cons_circ)
    prob = OptimizationProblem(optprob, x0, _p, ucons=[Inf], lcons=[0.0])

    sol = solve(prob, Ipopt.Optimizer())
    @test 10 * sol.minimum < l1

    sol = solve(prob, OptimizationMOI.MOI.OptimizerWithAttributes(Ipopt.Optimizer, "max_cpu_time" => 60.0))
    @test 10 * sol.minimum < l1

    nl_solver = OptimizationMOI.MOI.OptimizerWithAttributes(Ipopt.Optimizer, "print_level" => 0)
    minlp_solver = OptimizationMOI.MOI.OptimizerWithAttributes(Juniper.Optimizer, "nl_solver" => nl_solver)

    sol = solve(prob, minlp_solver)
    @test 10 * sol.minimum < l1

    sol = solve(prob, OptimizationMOI.MOI.OptimizerWithAttributes( () -> AmplNLWriter.Optimizer(Ipopt_jll.amplexe)))
end
