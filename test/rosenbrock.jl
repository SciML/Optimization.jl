using GalacticOptim, Optim, Test

rosenbrock(x, p) =  (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
_p  = [1.0, 100.0]

l1 = rosenbrock(x0, _p)
prob = OptimizationProblem(rosenbrock, x0, p=_p)
sol = solve(prob, SimulatedAnnealing())
@test 10*sol.minimum < l1

prob = OptimizationProblem(rosenbrock, x0, p=_p, lb=[-1.0, -1.0], ub=[0.8, 0.8])
sol = solve(prob, SAMIN())
@test 10*sol.minimum < l1

rosenbrock(x, p=nothing) =  (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2

l1 = rosenbrock(x0)
prob = OptimizationProblem(rosenbrock, x0)
sol = solve(prob, NelderMead()) 
@test 10*sol.minimum < l1


optprob = OptimizationFunction(rosenbrock, x0)

prob = OptimizationProblem(optprob, x0)
sol = solve(prob, BFGS())
@test 10*sol.minimum < l1

prob = OptimizationProblem(optprob, x0)
sol = solve(prob, Newton())
@test 10*sol.minimum < l1

prob = OptimizationProblem(optprob, x0)
sol = solve(prob, Optim.KrylovTrustRegion())
@test 10*sol.minimum < l1

prob = OptimizationProblem(optprob, x0, lb=[-1.0, -1.0], ub=[0.8, 0.8])
sol = solve(prob, Fminbox())
@test 10*sol.minimum < l1

prob = OptimizationProblem(optprob, x0, lb=[-1.0, -1.0], ub=[0.8, 0.8])
sol = solve(prob, SAMIN())
@test 10*sol.minimum < l1

using NLopt
prob = OptimizationProblem(optprob, x0)
sol = solve(prob, Opt(:LN_BOBYQA, 2))
@test 10*sol.minimum < l1

sol = solve(prob, Opt(:LD_LBFGS, 2))
@test 10*sol.minimum < l1

prob = OptimizationProblem(optprob, x0, lb=[-1.0, -1.0], ub=[0.8, 0.8])
sol = solve(prob, Opt(:G_MLSL_LDS, 2), nstart=5, local_method = Opt(:LD_LBFGS, 2))
@test 10*sol.minimum < l1

# using MultistartOptimization
# sol = solve(prob, MultistartOptimization.TikTak(100), local_method = NLopt.LD_LBFGS)
# @test 10*sol.minimum < l1

# using QuadDIRECT
# sol = solve(prob, QuadDirect(); splits = ([-0.5, 0.0, 0.5],[-0.5, 0.0, 0.5]))
# @test 10*sol.minimum < l1

# using Evolutionary
# sol = solve(prob, CMAES(μ = 5, λ = 100))
# @test 10*sol.minimum < l1

using BlackBoxOptim
prob = GalacticOptim.OptimizationProblem(optprob, x0, lb=[-1.0, -1.0], ub=[0.8, 0.8])
sol = solve(prob, BBO())
@test 10*sol.minimum < l1