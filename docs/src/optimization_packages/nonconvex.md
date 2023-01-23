# Nonconvex.jl

[`Nonconvex`](https://github.com/JuliaNonconvex/Nonconvex.jl) is a Julia package implementing and wrapping nonconvex constrained optimization algorithms.

## Installation: OptimizationNonconvex.jl

To use this package, install the OptimizationNonconvex package:

```julia
import Pkg;
Pkg.add("OptimizationNonconvex");
```

## Global Optimizer

### Without Constraint Equations

A `Nonconvex` algorithm is called using one of the following:

  - [Method of moving asymptotes (MMA)](https://julianonconvex.github.io/Nonconvex.jl/stable/algorithms/mma/#Method-of-moving-asymptotes-(MMA)):
    
      + `MMA87()`
      + `MMA02()`

  - [Ipopt](https://julianonconvex.github.io/Nonconvex.jl/stable/algorithms/ipopt/):
    
      + `IpoptAlg()`
  - [NLopt](https://julianonconvex.github.io/Nonconvex.jl/stable/algorithms/nlopt/):
    
      + `NLoptAlg(solver)` where solver can be any of the `NLopt` algorithms
  - [Augmented Lagrangian algorithm](https://julianonconvex.github.io/Nonconvex.jl/stable/algorithms/auglag/):
    
      + `AugLag()`
      + only works with constraints
  - [Mixed integer nonlinear programming (MINLP)](https://julianonconvex.github.io/Nonconvex.jl/stable/algorithms/minlp/):
    
      + Juniper + Ipopt: `JuniperIpoptAlg()`
      + Pavito + Ipopt + Cbc: `PavitoIpoptCbcAlg()`
  - [Multi-start optimization](https://julianonconvex.github.io/Nonconvex.jl/stable/algorithms/hyperopt/):
    
      + `HyperoptAlg(subsolver)` where `subalg` can be any of the described `Nonconvex` algorithm
  - [Surrogate-assisted Bayesian optimization](https://julianonconvex.github.io/Nonconvex.jl/stable/algorithms/surrogate/)
    
      + `BayesOptAlg(subsolver)` where `subalg` can be any of the described `Nonconvex` algorithm
  - [Multiple Trajectory Search](https://julianonconvex.github.io/Nonconvex.jl/stable/algorithms/mts/)
    
      + `MTSAlg()`

When performing optimizing a combination of integer and floating-point parameters, the `integer` keyword has to be set. It takes a boolean vector indicating which parameter is an integer.

## Notes

Some optimizer may require further options to be defined in order to work.

The currently available algorithms are listed [here](https://julianonconvex.github.io/Nonconvex.jl/stable/algorithms/algorithms/)

The algorithms in [`Nonconvex`](https://julianonconvex.github.io/Nonconvex.jl/stable/algorithms/algorithms/) are performing global optimization on problems without constraint equations. However, lower and upper constraints set by `lb` and `ub` in the `OptimizationProblem` are required.

## Examples

The Rosenbrock function can be optimized using the Method of moving asymptotes algorithm `MMA02()` as follows:

```julia
rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p = [1.0, 100.0]
f = OptimizationFunction(rosenbrock)
prob = Optimization.OptimizationProblem(f, x0, p, lb = [-1.0, -1.0], ub = [1.0, 1.0])
sol = solve(prob, MMA02(), maxiters = 100000, maxtime = 1000.0)
```

The options of for a sub-algorithm are passed simply as a NamedTuple and Optimization.jl infers the correct `Nonconvex` options struct:

```julia
rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p = [1.0, 100.0]
f = OptimizationFunction(rosenbrock)
prob = Optimization.OptimizationProblem(f, x0, p, lb = [-1.0, -1.0], ub = [1.0, 1.0])
sol = solve(prob, HyperoptAlg(IpoptAlg()), sub_options = (; max_iter = 100))
```

### With Constraint Equations

While `Nonconvex.jl` supports such constraints, `Optimization.jl` currently does not relay these constraints.
