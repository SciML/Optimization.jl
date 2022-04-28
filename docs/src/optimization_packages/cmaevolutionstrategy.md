# CMAEvolutionStrategy.jl
[`CMAEvolutionStrategy`](https://github.com/jbrea/CMAEvolutionStrategy.jl) is a Julia package implementing the **Covariance Matrix Adaptation Evolution Strategy algorithm**. 

The CMAEvolutionStrategy algorithm is called by `CMAEvolutionStrategyOpt()`

## Installation: GalacticCMAEvolutionStrategy.jl

To use this package, install the GalacticCMAEvolutionStrategy package:

```julia
import Pkg; Pkg.add("GalacticCMAEvolutionStrategy")
```

## Global Optimizer
### Without Constraint Equations

The method in [`CMAEvolutionStrategy`](https://github.com/jbrea/CMAEvolutionStrategy.jl) is performing global optimization on problems without
constraint equations. However, lower and upper constraints set by `lb` and `ub` in the `OptimizationProblem` are required.

## Example

The Rosenbrock function can optimized using the `CMAEvolutionStrategyOpt()` as follows:

```julia
rosenbrock(x, p) =  (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p  = [1.0, 100.0]
f = OptimizationFunction(rosenbrock)
prob = GalacticOptim.OptimizationProblem(f, x0, p, lb = [-1.0,-1.0], ub = [1.0,1.0])
sol = solve(prob, CMAEvolutionStrategyOpt())
```