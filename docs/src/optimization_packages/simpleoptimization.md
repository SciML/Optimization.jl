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
SimpleGradientDescent
SimpleNewton
SimpleSOAP
```

## Example

The Rosenbrock function can be optimized using `SimpleBFGS` as follows:

```@example SimpleOptimization
using SimpleOptimization, OptimizationBase, ForwardDiff
rosenbrock(x, p) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
x0 = zeros(2)
p = nothing
optf = OptimizationFunction(rosenbrock, OptimizationBase.AutoForwardDiff())
prob = OptimizationProblem(optf, x0, p)
sol = solve(prob, SimpleBFGS())
```

The same problem with `SimpleLBFGS` requires a statically sized `u0`:

```@example SimpleOptimization
using StaticArrays: SVector
x0s = SVector(0.0, 0.0)
optfs = OptimizationFunction{false}(rosenbrock, OptimizationBase.AutoForwardDiff())
probs = OptimizationProblem{false}(optfs, x0s, p)
sol = solve(probs, SimpleLBFGS())
prob_box = OptimizationProblem{false}(
    optfs, x0s, p; lb = SVector(-2.0, -2.0), ub = SVector(2.0, 2.0)
)
sol_box = solve(prob_box, SimpleLBFGS())
```
