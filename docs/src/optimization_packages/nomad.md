# NOMAD.jl
[`NOMAD`](https://github.com/bbopt/NOMAD.jl) is Julia package interfacing to NOMAD,
which is a C++ implementation of the Mesh Adaptive Direct Search algorithm (MADS),
designed for difficult blackbox optimization problems.
These issues occur when the functions defining the objective and constraints are the result of costly computer simulations.
[`NOMAD.jl documentation`](https://bbopt.github.io/NOMAD.jl/stable/)

The NOMAD algorithm is called by `NOMADOpt()`

## Installation: OptimizationNOMAD.jl

To use this package, install the OptimizationNOMAD package:

```julia
import Pkg; Pkg.add("OptimizationNOMAD")
```

## Global Optimizer
### Without Constraint Equations

The method in [`NOMAD`](https://github.com/bbopt/NOMAD.jl) is performing global optimization on problems both with and without
constraint equations. However, linear and nonlinear constraints defined in `Optimization` are currently not passed.

NOMAD works both with and without lower and upper box-constraints set by `lb` and `ub` in the `OptimizationProblem`.

## Examples

The Rosenbrock function can be optimized using the `NOMADOpt()` with and without box-constraints as follows:

```@example NOMAD
using Optimization, OptimizationNOMAD
rosenbrock(x, p) =  (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p  = [1.0, 100.0]
f = OptimizationFunction(rosenbrock)

prob = OptimizationProblem(f, x0, p)
sol = Optimization.solve(prob,NOMADOpt())

prob = OptimizationProblem(f, x0, p, lb = [-1.0,-1.0], ub = [1.5,1.5])
sol = Optimization.solve(prob,NOMADOpt())
```