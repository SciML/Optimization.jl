# GCMAES.jl
[`GCMAES`](https://github.com/AStupidBear/GCMAES.jl) is a Julia package implementing the **Gradient-based Covariance Matrix Adaptation Evolutionary Strategy** which can utilize the gradient information to speed up the optimization process.

## Installation: GalacticGCMAES.jl

To use this package, install the GalacticGCMAES package:

```julia
import Pkg; Pkg.add("GalacticGCMAES")
```

## Global Optimizer
### Without Constraint Equations

The GCMAES algorithm is called by `GCMAESOpt()` and the initial search variance is set as a keyword argument `σ0` (default: `σ0 = 0.2`)

The method in [`GCMAES`](https://github.com/AStupidBear/GCMAES.jl) is performing global optimization on problems without
constraint equations. However, lower and upper constraints set by `lb` and `ub` in the `OptimizationProblem` are required.

## Example

The Rosenbrock function can optimized using the `GCMAESOpt()` without utilizing the gradient information as follows:

```julia
rosenbrock(x, p) =  (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p  = [1.0, 100.0]
f = OptimizationFunction(rosenbrock)
prob = GalacticOptim.OptimizationProblem(f, x0, p, lb = [-1.0,-1.0], ub = [1.0,1.0])
sol = solve(prob, GCMAESOpt())
```

We can also utilise the gradient information of the optimization problem to aid the optimization as follows:

```julia
rosenbrock(x, p) =  (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p  = [1.0, 100.0]
f = OptimizationFunction(rosenbrock, GalacticOptim.ForwardDiff)
prob = GalacticOptim.OptimizationProblem(f, x0, p, lb = [-1.0,-1.0], ub = [1.0,1.0])
sol = solve(prob, GCMAESOpt())
```