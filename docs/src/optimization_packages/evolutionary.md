# Evolutionary.jl

[`Evolutionary`](https://github.com/wildart/Evolutionary.jl) is a Julia package implementing various evolutionary and genetic algorithm.

## Installation: OptimizationEvolutionary.jl

To use this package, install the OptimizationEvolutionary package:

```julia
import Pkg;
Pkg.add("OptimizationEvolutionary");
```

## Global Optimizer

### Without Constraint Equations

The methods in [`Evolutionary`](https://github.com/wildart/Evolutionary.jl) are performing global optimization on problems without
constraint equations. These methods work both with and without lower and upper constraints set by `lb` and `ub` in the `OptimizationProblem`.

A `Evolutionary` algorithm is called by one of the following:

  - [`Evolutionary.GA()`](https://wildart.github.io/Evolutionary.jl/stable/ga/): **Genetic Algorithm optimizer**

  - [`Evolutionary.DE()`](https://wildart.github.io/Evolutionary.jl/stable/de/): **Differential Evolution optimizer**
  - [`Evolutionary.ES()`](https://wildart.github.io/Evolutionary.jl/stable/es/): **Evolution Strategy algorithm**
  - [`Evolutionary.CMAES()`](https://wildart.github.io/Evolutionary.jl/stable/cmaes/): **Covariance Matrix Adaptation Evolution Strategy algorithm**

Algorithm-specific options are defined as `kwargs`. See the respective documentation for more detail.

## Example

The Rosenbrock function can be optimized using the `Evolutionary.CMAES()` as follows:

```@example Evolutionary
using Optimization, OptimizationEvolutionary
rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p = [1.0, 100.0]
f = OptimizationFunction(rosenbrock)
prob = Optimization.OptimizationProblem(f, x0, p, lb = [-1.0, -1.0], ub = [1.0, 1.0])
sol = solve(prob, Evolutionary.CMAES(μ = 40, λ = 100))
```

## Multi-objective optimization
The Rosenbrock and Ackley functions can be optimized using the `Evolutionary.NSGA2()` as follows:

```@example MOO-Evolutionary
using Optimization, OptimizationEvolutionary
function func(x, p=nothing)::Vector{Float64}
  f1 = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2  # Rosenbrock function
  f2 = -20.0 * exp(-0.2 * sqrt(0.5 * (x[1]^2 + x[2]^2))) - exp(0.5 * (cos(2π * x[1]) + cos(2π * x[2]))) + exp(1) + 20.0  # Ackley function
  return [f1, f2]
end
initial_guess = [1.0, 1.0]
function gradient_multi_objective(x, p=nothing)
    ForwardDiff.jacobian(func, x)
end
obj_func = MultiObjectiveOptimizationFunction(func, jac=gradient_multi_objective)
algorithm = OptimizationEvolutionary.NSGA2()
problem = OptimizationProblem(obj_func, initial_guess)
result = solve(problem, algorithm)
```
