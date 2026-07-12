# MadNLP.jl

`OptimizationMadNLP.jl` connects Optimization.jl problems to
[`MadNLP.jl`](https://github.com/MadNLP/MadNLP.jl), a nonlinear programming
solver for large-scale constrained optimization.

## Installation: OptimizationMadNLP.jl

```julia
import Pkg
Pkg.add("OptimizationMadNLP")
```

## Methods

```@docs
OptimizationMadNLP.MadNLPOptimizer
```

## Example

```julia
using Optimization, OptimizationMadNLP, ADTypes

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
optf = OptimizationFunction(rosenbrock, ADTypes.AutoForwardDiff())
prob = OptimizationProblem(optf, zeros(2), [1.0, 100.0])

sol = solve(prob, MadNLPOptimizer())
```
