# [Manopt.jl](@id manopt)
[`Manopt`](https://github.com/JuliaManifolds/Manopt.jl) is Julia package implementing various algorithm focusing on manifold-based constraints. When all or some of the constraints are in the form of Riemannian manifolds, optimization can be often performed more efficiently than with generic methods handling equality or inequality constraints.

See [`Manifolds`](https://github.com/JuliaManifolds/Manifolds.jl) for a library of manifolds that can be used as constraints.

## Installation: OptimizationManopt.jl

To use this package, install the `OptimizationManopt` package:

```julia
import Pkg; Pkg.add("OptimizationManopt")
```

## Methods

`Manopt.jl` algorithms can be accessed via Optimization.jl using one of the following optimizers:

- `GradientDescentOptimizer`
- `NelderMeadOptimizer`

For a more extensive documentation of all the algorithms and options please consult the [`Documentation`](https://manoptjl.org/stable/).

## Local Optimizer

### Local Constraint

### Derivative-Free

### Gradient-Based

## Global Optimizer

### Without Constraint Equations
