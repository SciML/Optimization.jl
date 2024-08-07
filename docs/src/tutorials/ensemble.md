# Multistart optimization with EnsembleProblem

The `EnsembleProblem` in SciML serves as a common interface for running a problem on multiple sets of initializations. In the context
of optimization, this is useful for performing multistart optimization.

This can be useful for complex, low dimensional problems. We demonstrate this, again, on the rosenbrock function.

```@example ensemble
using Optimization, OptimizationOptimJL, Random

Random.seed!(100)

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)

optf = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, x0, [1.0, 100.0])
@time sol1 = Optimization.solve(prob, OptimizationOptimJL.BFGS(), maxiters = 5)

@show sol1.objective

ensembleprob = Optimization.EnsembleProblem(
    prob, [x0, x0 .+ rand(2), x0 .+ rand(2), x0 .+ rand(2)])

@time sol = Optimization.solve(ensembleprob, OptimizationOptimJL.BFGS(),
    EnsembleThreads(), trajectories = 4, maxiters = 5)
@show findmin(i -> sol[i].objective, 1:4)[1]
```

With the same number of iterations (5) we get a much lower (1/100th) objective value by using multiple initial points. The initialization strategy used here was a pretty trivial one but approaches based on Quasi-Monte Carlo sampling should be typically more effective.
