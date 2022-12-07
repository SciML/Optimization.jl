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

- `ConjugateGradientDescentOptimizer`
- `GradientDescentOptimizer`
- `NelderMeadOptimizer`
- `ParticleSwarmOptimizer`
- `QuasiNewtonOptimizer`

For a more extensive documentation of all the algorithms and options please consult the [`Documentation`](https://manoptjl.org/stable/).

## Local Optimizer

### Derivative-Free

The Nelder-Mead optimizer can be used for local derivative-free optimization on manifolds.

```@example Manopt1
using Optimization, OptimizationManopt, Manifolds
rosenbrock(x, p) =  (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
x0 = [0.0, 1.0]
p  = [1.0,100.0]
f = OptimizationFunction(rosenbrock)
opt = OptimizationManopt.NelderMeadOptimizer(Sphere(1))
prob = Optimization.OptimizationProblem(f, x0, p)
sol = solve(prob, opt)
```

### Gradient-Based

Manopt offers gradient descent, conjugate gradient descent and quasi-Newton solvers for local gradient-based optimization.

Note that the returned gradient needs to be Riemannian, see for example [https://manoptjl.org/stable/functions/gradients/](https://manoptjl.org/stable/functions/gradients/).
Note that one way to obtain a Riemannian gradient is by [projection and (optional) Riesz representer change](https://juliamanifolds.github.io/Manifolds.jl/latest/features/differentiation.html#Manifolds.RiemannianProjectionBackend).

```@example Manopt2
using Optimization, OptimizationManopt, Manifolds
rosenbrock(x, p) =  (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
function rosenbrock_grad!(storage, x, p)
    storage[1] = -2.0 * (p[1] - x[1]) - 4.0 * p[2] * (x[2] - x[1]^2) * x[1]
    storage[2] = 2.0 * p[2] * (x[2] - x[1]^2)
    project!(Sphere(1), storage, x, storage)
end
x0 = [0.0, 1.0]
p  = [1.0,100.0]
f = OptimizationFunction(rosenbrock; grad = rosenbrock_grad!)
opt = OptimizationManopt.GradientDescentOptimizer(Sphere(1))
prob = Optimization.OptimizationProblem(f, x0, p)
sol = solve(prob, opt)
```

## Global Optimizer

### Without Constraint Equations

The particle swarm optimizer can be used for global optimization on a manifold without constraint equations. It can be especially useful on compact manifolds such as spheres or orthogonal matrices.

```@example Manopt3
using Optimization, OptimizationManopt, Manifolds
rosenbrock(x, p) =  (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
x0 = [0.0, 1.0]
p  = [1.0,100.0]
f = OptimizationFunction(rosenbrock)
opt = OptimizationManopt.ParticleSwarmOptimizer(Sphere(1))
prob = Optimization.OptimizationProblem(f, x0, p)
sol = solve(prob, opt)
```
