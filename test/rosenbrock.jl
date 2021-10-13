using MathOptInterface, GalacticOptim, Optim, Test, Random, Flux, ForwardDiff, Zygote
using Nonconvex

rosenbrock(x, p) =  (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
_p  = [1.0, 100.0]
l1 = rosenbrock(x0, _p)

@testset "Optim, CMAEvolutionStrategy" begin
    f = OptimizationFunction(rosenbrock, GalacticOptim.AutoForwardDiff())
    prob = OptimizationProblem(f, x0, _p)
    Random.seed!(1234)
    sol = solve(prob, SimulatedAnnealing())
    @test 10*sol.minimum < l1

    Random.seed!(1234)
    prob = OptimizationProblem(f, x0, _p, lb=[-1.0, -1.0], ub=[0.8, 0.8])
    sol = solve(prob, SAMIN())
    @test 10*sol.minimum < l1

    using CMAEvolutionStrategy
    sol = solve(prob, CMAEvolutionStrategyOpt())
    @test 10*sol.minimum < l1

    prob = OptimizationProblem(rosenbrock, x0, _p)
    sol = solve(prob, Optim.NelderMead())
    @test 10*sol.minimum < l1
end

@testset "Constrained Optimisation" begin
    cons= (x,p) -> [x[1]^2 + x[2]^2]
    optprob = OptimizationFunction(rosenbrock, GalacticOptim.AutoForwardDiff();cons= cons)

    prob = OptimizationProblem(optprob, x0, _p)

    sol = solve(prob, Flux.ADAM(0.1), maxiters = 1000)
    @test 10*sol.minimum < l1

    sol = solve(prob, Optim.BFGS())
    @test 10*sol.minimum < l1

    sol = solve(prob, Optim.Newton())
    @test 10*sol.minimum < l1

    sol = solve(prob, Optim.KrylovTrustRegion())
    @test 10*sol.minimum < l1

    prob = OptimizationProblem(optprob, x0, _p, lcons = [-Inf], ucons = [Inf])
    sol = solve(prob, IPNewton())
    @test 10*sol.minimum < l1

    prob = OptimizationProblem(optprob, x0, _p, lcons = [-5.0], ucons = [10.0])
    sol = solve(prob, IPNewton())
    @test 10*sol.minimum < l1

    prob = OptimizationProblem(optprob, x0, _p, lcons = [-Inf], ucons = [Inf], lb = [-500.0,-500.0], ub=[50.0,50.0])
    sol = solve(prob, IPNewton())
    @test sol.minimum < l1

    function con2_c(x,p)
        [x[1]^2 + x[2]^2, x[2]*sin(x[1])-x[1]]
    end

    optprob = OptimizationFunction(rosenbrock, GalacticOptim.AutoForwardDiff();cons= con2_c)
    prob = OptimizationProblem(optprob, x0, _p, lcons = [-Inf,-Inf], ucons = [Inf,Inf])
    sol = solve(prob, IPNewton())
    @test 10*sol.minimum < l1

    cons_circ = (x,p) -> [x[1]^2 + x[2]^2]
    optprob = OptimizationFunction(rosenbrock, GalacticOptim.AutoForwardDiff();cons= cons_circ)
    prob = OptimizationProblem(optprob, x0, _p, lcons = [-Inf], ucons = [0.25^2])
    sol = solve(prob, IPNewton())
    @test sqrt(cons(sol.u,nothing)[1]) ≈ 0.25 rtol = 1e-6

    optprob = OptimizationFunction(rosenbrock, GalacticOptim.AutoZygote())

    prob = OptimizationProblem(optprob, x0, _p)
    sol = solve(prob, Flux.ADAM(), maxiters = 1000, progress = false)
    @test 10*sol.minimum < l1

    prob = OptimizationProblem(optprob, x0, _p, lb=[-1.0, -1.0], ub=[0.8, 0.8])
    sol = solve(prob, Optim.Fminbox())
    @test 10*sol.minimum < l1

    Random.seed!(1234)
    prob = OptimizationProblem(optprob, x0, _p, lb=[-1.0, -1.0], ub=[0.8, 0.8])
    sol = solve(prob, Optim.SAMIN())
    @test 10*sol.minimum < l1
end

@testset "MaxSense test" begin
    optprob = OptimizationFunction((x,p) -> -rosenbrock(x,p), GalacticOptim.AutoZygote())
    prob = OptimizationProblem(optprob, x0, _p; sense = GalacticOptim.MaxSense)

    import Ipopt
    sol = solve(prob, Ipopt.Optimizer())
    @test 10*sol.minimum < l1

    import NLopt
    sol = solve(prob, NLopt.Opt(:LN_BOBYQA, 2))
    @test 10*sol.minimum < l1

    sol = solve(prob, NelderMead())
    @test 10*sol.minimum < l1

    sol = solve(prob, BFGS())
    @test 10*sol.minimum < l1

    function g!(G, x)
        G[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
        G[2] = 200.0 * (x[2] - x[1]^2)
    end
    optprob = OptimizationFunction((x,p) -> -rosenbrock(x,p), GalacticOptim.AutoZygote(), grad = g!)
    prob = OptimizationProblem(optprob, x0, _p; sense = GalacticOptim.MaxSense)
    sol = solve(prob, BFGS())
    @test 10*sol.minimum < l1
end

@testset "MinSense test" begin
    optprob = OptimizationFunction(rosenbrock, GalacticOptim.AutoZygote())
    prob = OptimizationProblem(optprob, x0, _p; sense = GalacticOptim.MinSense)

    sol = solve(prob, Ipopt.Optimizer())
    @test 10*sol.minimum < l1

    sol = solve(prob, GalacticOptim.MOI.OptimizerWithAttributes(Ipopt.Optimizer, "max_cpu_time" => 60.0))
    @test 10*sol.minimum < l1


    prob = OptimizationProblem(optprob, x0)
    sol = solve(prob, NLopt.Opt(:LN_BOBYQA, 2))
    @test 10*sol.minimum < l1

    sol = solve(prob, GalacticOptim.MOI.OptimizerWithAttributes(NLopt.Optimizer, "algorithm" => :LN_BOBYQA))
    @test 10*sol.minimum < l1

    sol = solve(prob, NLopt.Opt(:LD_LBFGS, 2))
    @test 10*sol.minimum < l1

    sol = solve(prob, GalacticOptim.MOI.OptimizerWithAttributes(NLopt.Optimizer, "algorithm" => :LD_LBFGS))
    @test 10*sol.minimum < l1

    prob = OptimizationProblem(optprob, x0, lb=[-1.0, -1.0], ub=[0.8, 0.8])
    sol = solve(prob, NLopt.Opt(:LD_LBFGS, 2))
    @test 10*sol.minimum < l1

    sol = solve(prob, GalacticOptim.MOI.OptimizerWithAttributes(NLopt.Optimizer, "algorithm" => :LD_LBFGS))
    @test 10*sol.minimum < l1

    sol = solve(prob, NLopt.Opt(:G_MLSL_LDS, 2), local_method = NLopt.Opt(:LD_LBFGS, 2), maxiters=10000)
    @test 10*sol.minimum < l1

    prob = OptimizationProblem(optprob, x0)
    sol = solve(prob, NLopt.LN_BOBYQA())
    @test 10*sol.minimum < l1

    sol = solve(prob, NLopt.LD_LBFGS())
    @test 10*sol.minimum < l1

    prob = OptimizationProblem(optprob, x0, lb=[-1.0, -1.0], ub=[0.8, 0.8])
    sol = solve(prob, NLopt.LD_LBFGS())
    @test 10*sol.minimum < l1

    sol = solve(prob, NLopt.G_MLSL_LDS(), local_method = NLopt.LD_LBFGS(), local_maxiters=10000, maxiters=10000, population=10)
    @test 10*sol.minimum < l1
end

# using MultistartOptimization
# sol = solve(prob, MultistartOptimization.TikTak(100), local_method = NLopt.LD_LBFGS())
# @test 10*sol.minimum < l1

# using QuadDIRECT
# sol = solve(prob, QuadDirect(); splits = ([-0.5, 0.0, 0.5],[-0.5, 0.0, 0.5]))
# @test 10*sol.minimum < l1

@testset "Evolutionary, BlackBoxOptim, Metaheuristics, Nonconvex, SpeedMapping" begin
    optprob = OptimizationFunction(rosenbrock, GalacticOptim.AutoZygote())
    using Evolutionary
    prob = GalacticOptim.OptimizationProblem(optprob, x0, _p)
    sol = solve(prob, CMAES(μ =40 , λ = 100),abstol=1e-15)
    @test 10*sol.minimum < l1

    using BlackBoxOptim
    prob = GalacticOptim.OptimizationProblem(optprob, x0, _p, lb=[-1.0, -1.0], ub=[0.8, 0.8])
    sol = solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited())
    @test 10*sol.minimum < l1

    using Metaheuristics
    prob = GalacticOptim.OptimizationProblem(optprob, x0, _p, lb=[-1.0, -1.0], ub=[1.5, 1.5])
    sol = solve(prob, ECA())
    @test 10*sol.minimum < l1

    sol = solve(prob, Metaheuristics.DE())
    @test 10*sol.minimum < l1

    sol = solve(prob, PSO())
    @test 10*sol.minimum < l1

    sol = solve(prob, ABC())
    @test 10*sol.minimum < l1

    sol = solve(prob, CGSA())
    @test 10*sol.minimum < l1

    sol = solve(prob, SA())
    @test 10*sol.minimum < l1

    sol = solve(prob, WOA())
    @test 10*sol.minimum < l1

    sol = solve(prob, ECA())
    @test 10*sol.minimum < l1

    sol = solve(prob, Metaheuristics.DE(), use_initial=true)
    @test 10*sol.minimum < l1

    sol = solve(prob, PSO(), use_initial=true)
    @test 10*sol.minimum < l1

    sol = solve(prob, ABC(), use_initial=true)
    @test 10*sol.minimum < l1

    sol = solve(prob, CGSA(), use_initial=true)
    @test 10*sol.minimum < l1

    sol = solve(prob, SA(), use_initial=true)
    @test 10*sol.minimum < l1

    sol = solve(prob, WOA(), use_initial=true)
    @test 10*sol.minimum < l1

    using ModelingToolkit
    f = OptimizationFunction(rosenbrock,ModelingToolkit.AutoModelingToolkit())
    prob = OptimizationProblem(f, x0, _p)
    @test_broken sol = solve(prob,Optim.Newton())


    ### Nonconvex test
    Nonconvex.@load MMA
    Nonconvex.@load Ipopt
    Nonconvex.@load NLopt
    Nonconvex.@load BayesOpt
    Nonconvex.@load Juniper
    Nonconvex.@load Pavito
    Nonconvex.@load Hyperopt
    Nonconvex.@load MTS
    prob = GalacticOptim.OptimizationProblem(optprob, x0, _p, lb = [-1.0,-1.0], ub = [1.5,1.5])

    sol = solve(prob, MMA02())
    @test 10*sol.minimum < l1

    sol = solve(prob, GCMMA())
    @test 10*sol.minimum < l1

    sol = solve(prob, IpoptAlg())
    @test 10*sol.minimum < l1

    sol = solve(prob, NLoptAlg(:LN_NELDERMEAD))
    @test 10*sol.minimum < l1

    sol = solve(prob, NLoptAlg(:LD_LBFGS))
    @test 10*sol.minimum < l1

    sol = solve(prob, MTSAlg())
    @test 10*sol.minimum < l1

    sol = solve(prob, HyperoptAlg(NLoptAlg(:LN_NELDERMEAD)))
    @test 10*sol.minimum < l1

    sol = solve(prob, HyperoptAlg(NLoptAlg(:LN_NELDERMEAD)), sampler=LHSampler())
    @test 10*sol.minimum < l1

    sol = solve(prob, HyperoptAlg(NLoptAlg(:LN_NELDERMEAD)), sampler=CLHSampler())
    @test 10*sol.minimum < l1

    sol = solve(prob, HyperoptAlg(NLoptAlg(:LN_NELDERMEAD)), sampler=GPSampler())
    @test 10*sol.minimum < l1

    sol = solve(prob, HyperoptAlg(IpoptAlg()))
    @test 10*sol.minimum < l1

    sol = solve(prob, HyperoptAlg(IpoptAlg()), sampler=LHSampler())
    @test 10*sol.minimum < l1

    sol = solve(prob, HyperoptAlg(IpoptAlg()), sampler=CLHSampler())
    @test 10*sol.minimum < l1

    sol = solve(prob, HyperoptAlg(IpoptAlg()), sampler=GPSampler())
    @test 10*sol.minimum < l1

    sol = solve(prob, BayesOptAlg(IpoptAlg()))
    @test 10*sol.minimum < l1

    sol = solve(prob, BayesOptAlg(IpoptAlg()))
    @test 10*sol.minimum < l1

    sol = solve(prob, BayesOptAlg(NLoptAlg(:LN_NELDERMEAD)))
    @test 10*sol.minimum < l1

    sol = solve(prob, BayesOptAlg(NLoptAlg(:LN_NELDERMEAD)))
    @test 10*sol.minimum < l1

    sol = solve(prob, BayesOptAlg(IpoptAlg()))
    @test 10*sol.minimum < l1

    sol = solve(prob, BayesOptAlg(IpoptAlg()))
    @test 10*sol.minimum < l1

    sol = solve(prob, BayesOptAlg(NLoptAlg(:LN_NELDERMEAD)))
    @test 10*sol.minimum < l1

    sol = solve(prob, BayesOptAlg(NLoptAlg(:LN_NELDERMEAD)))
    @test 10*sol.minimum < l1

    sol = solve(prob, MMA02(), maxiters=1000)
    @test 10*sol.minimum < l1

    sol = solve(prob, GCMMA(), maxiters=1000)
    @test 10*sol.minimum < l1

    sol = solve(prob, IpoptAlg(), maxiters=1000)
    @test 10*sol.minimum < l1

    sol = solve(prob, NLoptAlg(:LN_NELDERMEAD), maxiters=1000)
    @test 10*sol.minimum < l1

    sol = solve(prob, NLoptAlg(:LD_LBFGS), maxiters=1000)
    @test 10*sol.minimum < l1

    sol = solve(prob, MTSAlg(), maxiters=1000)
    @test 10*sol.minimum < l1

    sol = solve(prob, HyperoptAlg(NLoptAlg(:LN_NELDERMEAD)), maxiters=100)
    @test 10*sol.minimum < l1

    sol = solve(prob, HyperoptAlg(NLoptAlg(:LN_NELDERMEAD)), sampler=LHSampler(), maxiters=100)
    @test 10*sol.minimum < l1

    sol = solve(prob, HyperoptAlg(NLoptAlg(:LN_NELDERMEAD)), sampler=CLHSampler(), maxiters=100)
    @test 10*sol.minimum < l1

    sol = solve(prob, HyperoptAlg(NLoptAlg(:LN_NELDERMEAD)), sampler=GPSampler(), maxiters=100)
    @test 10*sol.minimum < l1

    sol = solve(prob, HyperoptAlg(IpoptAlg()), maxiters=100)
    @test 10*sol.minimum < l1

    sol = solve(prob, HyperoptAlg(IpoptAlg()), sampler=LHSampler(), maxiters=100)
    @test 10*sol.minimum < l1

    sol = solve(prob, HyperoptAlg(IpoptAlg()), sampler=CLHSampler(), maxiters=100)
    @test 10*sol.minimum < l1

    sol = solve(prob, HyperoptAlg(IpoptAlg()), sampler=GPSampler(), maxiters=100)
    @test 10*sol.minimum < l1

    sol = solve(prob, BayesOptAlg(IpoptAlg()), maxiters=1000)
    @test 10*sol.minimum < l1

    sol = solve(prob, BayesOptAlg(IpoptAlg()), maxiters=1000)
    @test 10*sol.minimum < l1

    sol = solve(prob, BayesOptAlg(NLoptAlg(:LN_NELDERMEAD)), maxiters=100)
    @test 10*sol.minimum < l1

    sol = solve(prob, BayesOptAlg(NLoptAlg(:LN_NELDERMEAD)), maxiters=100)
    @test 10*sol.minimum < l1

    sol = solve(prob, BayesOptAlg(IpoptAlg()), maxiters=100)
    @test 10*sol.minimum < l1

    sol = solve(prob, BayesOptAlg(IpoptAlg()), maxiters=100)
    @test 10*sol.minimum < l1

    sol = solve(prob, BayesOptAlg(NLoptAlg(:LN_NELDERMEAD)), maxiters=100)
    @test 10*sol.minimum < l1

    sol = solve(prob, BayesOptAlg(NLoptAlg(:LN_NELDERMEAD)), maxiters=100)
    @test 10*sol.minimum < l1

    sol = solve(prob, HyperoptAlg(NLoptAlg(:LN_NELDERMEAD)))
    @test 10*sol.minimum < l1

    ### suboptions
    sol = solve(prob, HyperoptAlg(NLoptAlg(:LN_NELDERMEAD)), sampler=LHSampler(), sub_options=(;maxeval=100))
    @test 10*sol.minimum < l1

    sol = solve(prob, HyperoptAlg(NLoptAlg(:LN_NELDERMEAD)), sampler=CLHSampler(), sub_options=(;maxeval=100))
    @test 10*sol.minimum < l1

    sol = solve(prob, HyperoptAlg(NLoptAlg(:LN_NELDERMEAD)), sampler=GPSampler(), sub_options=(;maxeval=100))
    @test 10*sol.minimum < l1

    sol = solve(prob, HyperoptAlg(IpoptAlg()), sub_options=(;max_iter=100))
    @test 10*sol.minimum < l1

    sol = solve(prob, HyperoptAlg(IpoptAlg()), sampler=LHSampler(), sub_options=(;max_iter=100))
    @test 10*sol.minimum < l1

    sol = solve(prob, HyperoptAlg(IpoptAlg()), sampler=CLHSampler(), sub_options=(;max_iter=100))
    @test 10*sol.minimum < l1

    sol = solve(prob, HyperoptAlg(IpoptAlg()), sampler=GPSampler(), sub_options=(;max_iter=100))
    @test 10*sol.minimum < l1

    sol = solve(prob, BayesOptAlg(IpoptAlg()), sub_options=(;max_iter=100))
    @test 10*sol.minimum < l1

    sol = solve(prob, BayesOptAlg(IpoptAlg()), sub_options=(;max_iter=100))
    @test 10*sol.minimum < l1

    sol = solve(prob, BayesOptAlg(NLoptAlg(:LN_NELDERMEAD)), sub_options=(;maxeval=100))
    @test 10*sol.minimum < l1

    sol = solve(prob, BayesOptAlg(NLoptAlg(:LN_NELDERMEAD)), sub_options=(;maxeval=100))
    @test 10*sol.minimum < l1

    sol = solve(prob, BayesOptAlg(IpoptAlg()), sub_options=(;max_iter=100))
    @test 10*sol.minimum < l1

    sol = solve(prob, BayesOptAlg(IpoptAlg()), sub_options=(;max_iter=100))
    @test 10*sol.minimum < l1

    sol = solve(prob, BayesOptAlg(NLoptAlg(:LN_NELDERMEAD)), sub_options=(;maxeval=100))
    @test 10*sol.minimum < l1

    sol = solve(prob, BayesOptAlg(NLoptAlg(:LN_NELDERMEAD)), sub_options=(;maxeval=100))
    @test 10*sol.minimum < l1

    using SpeedMapping
    f = OptimizationFunction(rosenbrock, GalacticOptim.AutoForwardDiff())
    prob = OptimizationProblem(f, x0, _p)
    sol = solve(prob,SpeedMappingOpt())

    prob = OptimizationProblem(f, x0, _p;lb=[0.0,0.0], ub=[1.0,1.0])
    sol = solve(prob,SpeedMappingOpt())

    f = OptimizationFunction(rosenbrock)
    prob = OptimizationProblem(f, x0, _p)
    sol = solve(prob,SpeedMappingOpt())

    prob = OptimizationProblem(f, x0, _p;lb=[0.0,0.0], ub=[1.0,1.0])
    sol = solve(prob,SpeedMappingOpt())
end
