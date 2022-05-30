# BlackBoxOptim.jl
[`BlackBoxOptim`](https://github.com/robertfeldt/BlackBoxOptim.jl) is a is a Julia package implementing **(Meta-)heuristic/stochastic algorithms** that do not require for the optimized function to be differentiable.

## Installation: OptimizationBBO.jl

To use this package, install the OptimizationBBO package:

```julia
import Pkg; Pkg.add("OptimizationBBO")
```

## Global Optimizers

### Without Constraint Equations

The algorithms in [`BlackBoxOptim`](https://github.com/robertfeldt/BlackBoxOptim.jl) are performing global optimization on problems without
constraint equations. However, lower and upper constraints set by `lb` and `ub` in the `OptimizationProblem` are required.

A `BlackBoxOptim` algorithm is called by `BBO_` prefix followed by the algorithm name:

* Natural Evolution Strategies:
  - Separable NES: `BBO_separable_nes()`
  - Exponential NES: `BBO_xnes()`
  - Distance-weighted Exponential NES: `BBO_dxnes()`
* Differential Evolution optimizers, 5 different:
  - Adaptive DE/rand/1/bin: `BBO_adaptive_de_rand_1_bin()`
  - Adaptive DE/rand/1/bin with radius limited sampling: `BBO_adaptive_de_rand_1_bin_radiuslimited()`
  - DE/rand/1/bin: `BBO_de_rand_1_bin()`
  - DE/rand/1/bin with radius limited sampling (a type of trivial geography): `BBO_de_rand_1_bin_radiuslimited()`
  - DE/rand/2/bin: `de_rand_2_bin()`
  - DE/rand/2/bin with radius limited sampling (a type of trivial geography): `BBO_de_rand_2_bin_radiuslimited()`
* Direct search:
  - Generating set search:
    - Compass/coordinate search: `BBO_generating_set_search()`
    - Direct search through probabilistic descent: `BBO_probabilistic_descent()`
* Resampling Memetic Searchers:
  - Resampling Memetic Search (RS): `BBO_resampling_memetic_search()`
  - Resampling Inheritance Memetic Search (RIS): `BBO_resampling_inheritance_memetic_search()`
* Stochastic Approximation:
  - Simultaneous Perturbation Stochastic Approximation (SPSA): `BBO_simultaneous_perturbation_stochastic_approximation()`
* RandomSearch (to compare to): `BBO_random_search()`

The recommended optimizer is `BBO_adaptive_de_rand_1_bin_radiuslimited()`

The currently available algorithms are listed [here](https://github.com/robertfeldt/BlackBoxOptim.jl#state-of-the-library)

## Example

The Rosenbrock function can optimized using the `BBO_adaptive_de_rand_1_bin_radiuslimited()` as follows:

```julia
rosenbrock(x, p) =  (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p  = [1.0, 100.0]
f = OptimizationFunction(rosenbrock)
prob = Optimization.OptimizationProblem(f, x0, p, lb = [-1.0,-1.0], ub = [1.0,1.0])
sol = solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxiters=100000, maxtime=1000.0)
```




