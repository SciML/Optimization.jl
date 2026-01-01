# [SimpleOptimization.jl](@id simpleoptimization)

[SimpleOptimization.jl](https://github.com/SciML/SimpleOptimization.jl) provides lightweight loop-unrolled optimization algorithms for the SciML ecosystem. It is designed for small-scale optimization problems where low overhead is critical.

## Installation: SimpleOptimization.jl

To use this package, install the SimpleOptimization package:

```julia
import Pkg;
Pkg.add("SimpleOptimization");
```

## Methods

```@docs
SimpleBFGS
SimpleLBFGS
```

## Example

The Rosenbrock function can be optimized using `SimpleBFGS` as follows:

```@example SimpleOptimization
using SimpleOptimization, Optimization, ForwardDiff
rosenbrock(x, p) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
x0 = zeros(2)
p = nothing
optf = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, x0, p)
sol = solve(prob, SimpleBFGS())
```
