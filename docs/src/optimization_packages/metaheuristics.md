# Metaheuristics.jl
[`Metaheuristics`](https://github.com/jmejia8/Metaheuristics.jl) is a is a Julia package implementing **metaheuristic algorithms** for global optiimization that do not require for the optimized function to be differentiable.

## Installation: OptimizationMetaheuristics.jl

To use this package, install the OptimizationMetaheuristics package:

```julia
import Pkg; Pkg.add("OptimizationMetaheuristics")
```

## Global Optimizer
### Without Constraint Equations

A `Metaheuristics` Single-Objective algorithm is called using one of the following:

* Evolutionary Centers Algorithm: `ECA()`
* Differential Evolution: `DE()` with 5 different stratgies
  - `DE(strategy=:rand1)` - default strategy
  - `DE(strategy=:rand2)`
  - `DE(strategy=:best1)`
  - `DE(strategy=:best2)`
  - `DE(strategy=:randToBest1)`
* Particle Swarm Optimization: `PSO()`
* Artificial Bee Colony: `ABC()`
* Gravitational Search Algorithm: `CGSA()`
* Simulated Annealing: `SA()`
* Whale Optimization Algorithm: `WOA()`

`Metaheuristics` also performs [`Multiobjective optimization`](https://jmejia8.github.io/Metaheuristics.jl/stable/examples/#Multiobjective-Optimization) but this is not yet supported by `Optimization`.

Each optimizer sets default settings based on the optimization problem but specific parameters can be set as shown in the original [`Documentation`](https://jmejia8.github.io/Metaheuristics.jl/stable/algorithms/) 

Additionally, `Metaheuristics` common settings which would be defined by [`Metaheuristics.Options`](https://jmejia8.github.io/Metaheuristics.jl/stable/api/#Metaheuristics.Options) can be simply passed as special keywoard arguments to `solve` without the need to use the `Metaheuristics.Options` struct.

Lastly, information about the optimization problem such as the true optimum is set via [`Metaheuristics.Information`](https://jmejia8.github.io/Metaheuristics.jl/stable/api/#Metaheuristics.Information) and passed as part of the optimizer struct to `solve` e.g. `solve(prob, ECA(information=Metaheuristics.Inoformation(f_optimum = 0.0)))`



The currently available algorithms and their parameters are listed [here](https://jmejia8.github.io/Metaheuristics.jl/stable/algorithms/).

## Notes

The algorithms in [`Metaheuristics`](https://github.com/jmejia8/Metaheuristics.jl) are performing global optimization on problems without
constraint equations. However, lower and upper constraints set by `lb` and `ub` in the `OptimizationProblem` are required.

## Examples

The Rosenbrock function can optimized using the Evolutionary Centers Algorithm `ECA()` as follows:

```julia
rosenbrock(x, p) =  (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p  = [1.0, 100.0]
f = OptimizationFunction(rosenbrock)
prob = Optimization.OptimizationProblem(f, x0, p, lb = [-1.0,-1.0], ub = [1.0,1.0])
sol = solve(prob, ECA(), maxiters=100000, maxtime=1000.0)
```

Per default `Metaheuristics` ignores the initial values `x0` set in the `OptimizationProblem`. In order to for `Optimization` to use `x0` we have to set `use_initial=true`:

```julia
rosenbrock(x, p) =  (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p  = [1.0, 100.0]
f = OptimizationFunction(rosenbrock)
prob = Optimization.OptimizationProblem(f, x0, p, lb = [-1.0,-1.0], ub = [1.0,1.0])
sol = solve(prob, ECA(), use_initial=true, maxiters=100000, maxtime=1000.0)
```




### With Constraint Equations

While `Metaheuristics.jl` supports such constraints, `Optimization.jl` currently does not relay these constraints.



