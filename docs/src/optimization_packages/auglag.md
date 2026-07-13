# AugLag.jl

`OptimizationAuglag.jl` provides an augmented Lagrangian wrapper for constrained
Optimization.jl problems. It repeatedly solves augmented subproblems with a user
selected inner optimizer.

## Installation: OptimizationAuglag.jl

```julia
import Pkg
Pkg.add("OptimizationAuglag")
```

## Methods

```@docs
OptimizationAuglag.AugLag
```

## Example

```julia
using Optimization, OptimizationAuglag, OptimizationOptimJL, ADTypes

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
cons(res, x, p) = (res .= [x[1]^2 + x[2]^2])

optf = OptimizationFunction(rosenbrock, ADTypes.AutoForwardDiff(); cons)
prob = OptimizationProblem(
    optf, zeros(2), [1.0, 100.0];
    lcons = [1.0], ucons = [1.0],
)

sol = solve(prob, AugLag(inner = BFGS()); maxiters = 100)
```
