# SpeedMapping.jl

[`SpeedMapping`](https://github.com/NicolasL-S/SpeedMapping.jl) accelerates the convergence of a mapping to a fixed point by the Alternating cyclic extrapolation algorithm which can also perform multivariate optimization based on the gradient function.

The SpeedMapping algorithm is called by `SpeedMappingOpt()`

## Installation: OptimizationSpeedMapping.jl

To use this package, install the OptimizationSpeedMapping package:

```julia
import Pkg;
Pkg.add("OptimizationSpeedMapping");
```

## Global Optimizer

### Without Constraint Equations

The method in [`SpeedMapping`](https://github.com/NicolasL-S/SpeedMapping.jl) is performing optimization on problems without
constraint equations. Lower and upper constraints set by `lb` and `ub` in the `OptimizationProblem` are optional.

If no AD backend is defined via `OptimizationFunction` the gradient is calculated via `SpeedMapping`'s ForwardDiff AD backend.

The Rosenbrock function can be optimized using the `SpeedMappingOpt()` with and without bound as follows:

```@example SpeedMapping
using Optimization, OptimizationSpeedMapping, ADTypes, ForwardDiff
rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p = [1.0, 100.0]
f = OptimizationFunction(rosenbrock, ADTypes.AutoForwardDiff())
prob = OptimizationProblem(f, x0, p)
sol = solve(prob, SpeedMappingOpt())

prob = OptimizationProblem(f, x0, p; lb = [0.0, 0.0], ub = [1.0, 1.0])
sol = solve(prob, SpeedMappingOpt())
```
