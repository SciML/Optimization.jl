using Optimization, OptimizationOptimJL, ForwardDiff

x0 = zeros(2)
rosenbrock(x, p = nothing) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
l1 = rosenbrock(x0)

optf = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, x0)
sol1 = Optimization.solve(prob, OptimizationOptimJL.BFGS(), maxiters = 5)


ensembleprob = Optimization.EnsembleProblem(prob, [x0, x0 .+ rand(2), x0 .+ rand(2), x0 .+ rand(2)])
sol = Optimization.solve(ensembleprob, OptimizationOptimJL.BFGS(), trajectories = 4, maxiters = 5)

@test findmin(i -> sol[i].objective, 1:4) < sol1.objective