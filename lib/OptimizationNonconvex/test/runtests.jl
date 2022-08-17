using OptimizationNonconvex, Optimization, Zygote, Pkg
using Test

@testset "OptimizationNonconvex.jl" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 100.0]
    l1 = rosenbrock(x0, _p)
    optprob = OptimizationFunction(rosenbrock, Optimization.AutoZygote())

    OptimizationNonconvex.Nonconvex.@load MMA
    OptimizationNonconvex.Nonconvex.@load Ipopt
    OptimizationNonconvex.Nonconvex.@load NLopt
    OptimizationNonconvex.Nonconvex.@load BayesOpt
    OptimizationNonconvex.Nonconvex.@load Juniper
    OptimizationNonconvex.Nonconvex.@load Pavito
    OptimizationNonconvex.Nonconvex.@load Hyperopt
    OptimizationNonconvex.Nonconvex.@load MTS
    prob = Optimization.OptimizationProblem(optprob, x0, _p, lb = [-1.0, -1.0],
                                            ub = [1.5, 1.5])

    sol = solve(prob, MMA02())
    @test 10 * sol.minimum < l1

    sol = solve(prob, GCMMA())
    @test 10 * sol.minimum < l1

    sol = solve(prob, IpoptAlg())
    @test 10 * sol.minimum < l1

    sol = solve(prob, NLoptAlg(:LN_NELDERMEAD))
    @test 10 * sol.minimum < l1

    sol = solve(prob, NLoptAlg(:LD_LBFGS))
    @test 10 * sol.minimum < l1

    sol = solve(prob, MTSAlg())
    @test 10 * sol.minimum < l1

    sol = solve(prob, HyperoptAlg(NLoptAlg(:LN_NELDERMEAD)))
    @test 10 * sol.minimum < l1

    sol = solve(prob, HyperoptAlg(NLoptAlg(:LN_NELDERMEAD)), sampler = LHSampler())
    @test 10 * sol.minimum < l1

    sol = solve(prob, HyperoptAlg(NLoptAlg(:LN_NELDERMEAD)), sampler = CLHSampler())
    @test 10 * sol.minimum < l1

    sol = solve(prob, HyperoptAlg(NLoptAlg(:LN_NELDERMEAD)), sampler = GPSampler())
    @test 10 * sol.minimum < l1

    sol = solve(prob, HyperoptAlg(IpoptAlg()))
    @test 10 * sol.minimum < l1

    sol = solve(prob, HyperoptAlg(IpoptAlg()), sampler = LHSampler())
    @test 10 * sol.minimum < l1

    sol = solve(prob, HyperoptAlg(IpoptAlg()), sampler = CLHSampler())
    @test 10 * sol.minimum < l1

    sol = solve(prob, HyperoptAlg(IpoptAlg()), sampler = GPSampler())
    @test 10 * sol.minimum < l1

    sol = solve(prob, BayesOptAlg(IpoptAlg()))
    @test 10 * sol.minimum < l1

    sol = solve(prob, BayesOptAlg(IpoptAlg()))
    @test 10 * sol.minimum < l1

    sol = solve(prob, BayesOptAlg(NLoptAlg(:LN_NELDERMEAD)))
    @test 10 * sol.minimum < l1

    sol = solve(prob, BayesOptAlg(NLoptAlg(:LN_NELDERMEAD)))
    @test 10 * sol.minimum < l1

    sol = solve(prob, BayesOptAlg(IpoptAlg()))
    @test 10 * sol.minimum < l1

    sol = solve(prob, BayesOptAlg(IpoptAlg()))
    @test 10 * sol.minimum < l1

    sol = solve(prob, BayesOptAlg(NLoptAlg(:LN_NELDERMEAD)))
    @test 10 * sol.minimum < l1

    sol = solve(prob, BayesOptAlg(NLoptAlg(:LN_NELDERMEAD)))
    @test 10 * sol.minimum < l1

    sol = solve(prob, MMA02(), maxiters = 1000)
    @test 10 * sol.minimum < l1

    sol = solve(prob, GCMMA(), maxiters = 1000)
    @test 10 * sol.minimum < l1

    sol = solve(prob, IpoptAlg(), maxiters = 1000)
    @test 10 * sol.minimum < l1

    sol = solve(prob, NLoptAlg(:LN_NELDERMEAD), maxiters = 1000)
    @test 10 * sol.minimum < l1

    sol = solve(prob, NLoptAlg(:LD_LBFGS), maxiters = 1000)
    @test 10 * sol.minimum < l1

    sol = solve(prob, MTSAlg(), maxiters = 1000)
    @test 10 * sol.minimum < l1

    sol = solve(prob, HyperoptAlg(NLoptAlg(:LN_NELDERMEAD)), maxiters = 100)
    @test 10 * sol.minimum < l1

    sol = solve(prob, HyperoptAlg(NLoptAlg(:LN_NELDERMEAD)), sampler = LHSampler(),
                maxiters = 100)
    @test 10 * sol.minimum < l1

    sol = solve(prob, HyperoptAlg(NLoptAlg(:LN_NELDERMEAD)), sampler = CLHSampler(),
                maxiters = 100)
    @test 10 * sol.minimum < l1

    sol = solve(prob, HyperoptAlg(NLoptAlg(:LN_NELDERMEAD)), sampler = GPSampler(),
                maxiters = 100)
    @test 10 * sol.minimum < l1

    sol = solve(prob, HyperoptAlg(IpoptAlg()), maxiters = 100)
    @test 10 * sol.minimum < l1

    sol = solve(prob, HyperoptAlg(IpoptAlg()), sampler = LHSampler(), maxiters = 100)
    @test 10 * sol.minimum < l1

    sol = solve(prob, HyperoptAlg(IpoptAlg()), sampler = CLHSampler(), maxiters = 100)
    @test 10 * sol.minimum < l1

    sol = solve(prob, HyperoptAlg(IpoptAlg()), sampler = GPSampler(), maxiters = 100)
    @test 10 * sol.minimum < l1

    sol = solve(prob, BayesOptAlg(IpoptAlg()), maxiters = 1000)
    @test 10 * sol.minimum < l1

    sol = solve(prob, BayesOptAlg(IpoptAlg()), maxiters = 1000)
    @test 10 * sol.minimum < l1

    sol = solve(prob, BayesOptAlg(NLoptAlg(:LN_NELDERMEAD)), maxiters = 100)
    @test 10 * sol.minimum < l1

    sol = solve(prob, BayesOptAlg(NLoptAlg(:LN_NELDERMEAD)), maxiters = 100)
    @test 10 * sol.minimum < l1

    sol = solve(prob, BayesOptAlg(IpoptAlg()), maxiters = 100)
    @test 10 * sol.minimum < l1

    sol = solve(prob, BayesOptAlg(IpoptAlg()), maxiters = 100)
    @test 10 * sol.minimum < l1

    sol = solve(prob, BayesOptAlg(NLoptAlg(:LN_NELDERMEAD)), maxiters = 100)
    @test 10 * sol.minimum < l1

    sol = solve(prob, BayesOptAlg(NLoptAlg(:LN_NELDERMEAD)), maxiters = 100)
    @test 10 * sol.minimum < l1

    sol = solve(prob, HyperoptAlg(NLoptAlg(:LN_NELDERMEAD)))
    @test 10 * sol.minimum < l1

    ### suboptions
    sol = solve(prob, HyperoptAlg(NLoptAlg(:LN_NELDERMEAD)), sampler = LHSampler(),
                sub_options = (; maxeval = 100))
    @test 10 * sol.minimum < l1

    sol = solve(prob, HyperoptAlg(NLoptAlg(:LN_NELDERMEAD)), sampler = CLHSampler(),
                sub_options = (; maxeval = 100))
    @test 10 * sol.minimum < l1

    sol = solve(prob, HyperoptAlg(NLoptAlg(:LN_NELDERMEAD)), sampler = GPSampler(),
                sub_options = (; maxeval = 100))
    @test 10 * sol.minimum < l1

    sol = solve(prob, HyperoptAlg(IpoptAlg()), sub_options = (; max_iter = 100))
    @test 10 * sol.minimum < l1

    sol = solve(prob, HyperoptAlg(IpoptAlg()), sampler = LHSampler(),
                sub_options = (; max_iter = 100))
    @test 10 * sol.minimum < l1

    sol = solve(prob, HyperoptAlg(IpoptAlg()), sampler = CLHSampler(),
                sub_options = (; max_iter = 100))
    @test 10 * sol.minimum < l1

    sol = solve(prob, HyperoptAlg(IpoptAlg()), sampler = GPSampler(),
                sub_options = (; max_iter = 100))
    @test 10 * sol.minimum < l1

    sol = solve(prob, BayesOptAlg(IpoptAlg()), sub_options = (; max_iter = 100))
    @test 10 * sol.minimum < l1

    sol = solve(prob, BayesOptAlg(IpoptAlg()), sub_options = (; max_iter = 100))
    @test 10 * sol.minimum < l1

    sol = solve(prob, BayesOptAlg(NLoptAlg(:LN_NELDERMEAD)),
                sub_options = (; maxeval = 100))
    @test 10 * sol.minimum < l1

    sol = solve(prob, BayesOptAlg(NLoptAlg(:LN_NELDERMEAD)),
                sub_options = (; maxeval = 100))
    @test 10 * sol.minimum < l1

    sol = solve(prob, BayesOptAlg(IpoptAlg()), sub_options = (; max_iter = 100))
    @test 10 * sol.minimum < l1

    sol = solve(prob, BayesOptAlg(IpoptAlg()), sub_options = (; max_iter = 100))
    @test 10 * sol.minimum < l1

    sol = solve(prob, BayesOptAlg(NLoptAlg(:LN_NELDERMEAD)),
                sub_options = (; maxeval = 100))
    @test 10 * sol.minimum < l1

    sol = solve(prob, BayesOptAlg(NLoptAlg(:LN_NELDERMEAD)),
                sub_options = (; maxeval = 100))
    @test 10 * sol.minimum < l1
end
