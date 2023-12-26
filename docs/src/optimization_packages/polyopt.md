# OptimizationPolyalgorithms.jl

OptimizationPolyalgorithms.jl is a package for collecting polyalgorithms formed by fusing popular optimization solvers of different characteristics.

## Installation: OptimizationPolyalgorithms

To use this package, install the OptimizationPolyalgorithms package:

```julia
import Pkg;
Pkg.add("OptimizationPolyalgorithms");
```

## Algorithms

Right now we support the following polyalgorithms.

`PolyOpt`: Runs Adam followed by BFGS for an equal number of iterations. This is useful in scientific machine learning use cases, by exploring the loss surface with the stochastic optimizer and converging to the minima faster with BFGS.

```@example polyopt
using Optimization, OptimizationPolyalgorithms
rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
_p = [1.0, 100.0]

optprob = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optprob, x0, _p)
sol = Optimization.solve(prob, PolyOpt(), maxiters = 1000)
```
