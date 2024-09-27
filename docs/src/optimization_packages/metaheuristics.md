# Metaheuristics.jl

[`Metaheuristics`](https://github.com/jmejia8/Metaheuristics.jl) is a Julia package implementing **metaheuristic algorithms** for global optimization that does not require for the optimized function to be differentiable.

## Installation: OptimizationMetaheuristics.jl

To use this package, install the OptimizationMetaheuristics package:

```julia
import Pkg;
Pkg.add("OptimizationMetaheuristics");
```

## Global Optimizer

### Without Constraint Equations

A `Metaheuristics` Single-Objective algorithm is called using one of the following:

  - Evolutionary Centers Algorithm: `ECA()`

  - Differential Evolution: `DE()` with 5 different strategies
    
      + `DE(strategy=:rand1)` - default strategy
      + `DE(strategy=:rand2)`
      + `DE(strategy=:best1)`
      + `DE(strategy=:best2)`
      + `DE(strategy=:randToBest1)`
  - Particle Swarm Optimization: `PSO()`
  - Artificial Bee Colony: `ABC()`
  - Gravitational Search Algorithm: `CGSA()`
  - Simulated Annealing: `SA()`
  - Whale Optimization Algorithm: `WOA()`

`Metaheuristics` also performs [`Multiobjective optimization`](https://jmejia8.github.io/Metaheuristics.jl/stable/examples/#Multiobjective-Optimization), but this is not yet supported by `Optimization`.

Each optimizer sets default settings based on the optimization problem, but specific parameters can be set as shown in the original [`Documentation`](https://jmejia8.github.io/Metaheuristics.jl/stable/algorithms/)

Additionally, `Metaheuristics` common settings which would be defined by [`Metaheuristics.Options`](https://jmejia8.github.io/Metaheuristics.jl/stable/api/#Metaheuristics.Options) can be simply passed as special keyword arguments to `solve` without the need to use the `Metaheuristics.Options` struct.

Lastly, information about the optimization problem such as the true optimum is set via [`Metaheuristics.Information`](https://jmejia8.github.io/Metaheuristics.jl/stable/api/#Metaheuristics.Information) and passed as part of the optimizer struct to `solve` e.g., `solve(prob, ECA(information=Metaheuristics.Information(f_optimum = 0.0)))`

The currently available algorithms and their parameters are listed [here](https://jmejia8.github.io/Metaheuristics.jl/stable/algorithms/).

## Notes

The algorithms in [`Metaheuristics`](https://github.com/jmejia8/Metaheuristics.jl) are performing global optimization on problems without
constraint equations. However, lower and upper constraints set by `lb` and `ub` in the `OptimizationProblem` are required.

## Examples

The Rosenbrock function can be optimized using the Evolutionary Centers Algorithm `ECA()` as follows:

```@example Metaheuristics
using Optimization, OptimizationMetaheuristics
rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p = [1.0, 100.0]
f = OptimizationFunction(rosenbrock)
prob = Optimization.OptimizationProblem(f, x0, p, lb = [-1.0, -1.0], ub = [1.0, 1.0])
sol = solve(prob, ECA(), maxiters = 100000, maxtime = 1000.0)
```

Per default `Metaheuristics` ignores the initial values `x0` set in the `OptimizationProblem`. In order to for `Optimization` to use `x0` we have to set `use_initial=true`:

```@example Metaheuristics
sol = solve(prob, ECA(), use_initial = true, maxiters = 100000, maxtime = 1000.0)
```

### With Constraint Equations

While `Metaheuristics.jl` supports such constraints, `Optimization.jl` currently does not relay these constraints.


## Multi-objective optimization
The zdt1 functions can be optimized using the `Metaheuristics.jl` as follows:

```@example MOO-Metaheuristics
using Optimization, OptimizationEvolutionary,OptimizationMetaheuristics, Metaheuristics
function zdt1(x)
    f1 = x[1]
    g = 1 + 9 * mean(x[2:end])
    h = 1 - sqrt(f1 / g)
    f2 = g * h
    # In this example, we have no constraints
    gx = [0.0]  # Inequality constraints (not used)
    hx = [0.0]  # Equality constraints (not used)
    return [f1, f2], gx, hx
end
multi_obj_fun = MultiObjectiveOptimizationFunction((x, p) -> zdt1(x))

# Define the problem bounds
lower_bounds = [0.0, 0.0, 0.0]
upper_bounds = [1.0, 1.0, 1.0]

# Define the initial guess
initial_guess = [0.5, 0.5, 0.5]

# Create the optimization problem
prob = OptimizationProblem(multi_obj_fun, initial_guess; lb = lower_bounds, ub = upper_bounds)

nobjectives = 2
npartitions = 100

# reference points (Das and Dennis's method)
weights = Metaheuristics.gen_ref_dirs(nobjectives, npartitions)

# Choose the algorithm as required.
alg1 = NSGA2()
alg2 = NSGA3()
alg3 = SPEA2()
alg4 = CCMO(NSGA2(N=100, p_m=0.001))
alg5 = MOEAD_DE(weights, options=Options(debug=false, iterations = 250))
alg6 = SMS_EMOA()

# Solve the problem
sol1 = solve(prob, alg1; maxiters = 100, use_initial = true)
sol2 = solve(prob, alg2; maxiters = 100, use_initial = true)
sol3 = solve(prob, alg3; maxiters = 100, use_initial = true)
sol4 = solve(prob, alg4)
sol5 = solve(prob, alg5; maxiters = 100, use_initial = true)
sol6 = solve(prob, alg6; maxiters = 100, use_initial = true)
```
