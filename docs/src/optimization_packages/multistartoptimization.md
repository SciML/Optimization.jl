# MultiStartOptimization.jl
[`MultistartOptimization`](https://github.com/tpapp/MultistartOptimization.jl) is a is a Julia package implementing a global optimization multistart method which performs local optimization after choosing multiple starting points.

`MultistartOptimization` requires both a global and local method to be defined. The global multistart method chooses a set of initial starting points from where local the local method starts from.

Currently, only one global method (`TikTak`) is implemented and called by `MultiStartOptimization.TikTak(n)` where `n` is the number of initial Sobol points. 

## Installation: OptimizationMultiStartOptimization.jl

To use this package, install the OptimizationMultiStartOptimization package:

```julia
import Pkg; Pkg.add("OptimizationMultiStartOptimization")
```
!!! note

You also need to load the relevant subpackage for the local method of you choice, for example if you plan to use one of the NLopt.jl's optimizers, you'd install and load OptimizationNLopt as described in the [NLopt.jl](@ref)'s section.

## Global Optimizer
### Without Constraint Equations

The methods in [`MultistartOptimization`](https://github.com/tpapp/MultistartOptimization.jl) is performing global optimization on problems without
constraint equations. However, lower and upper constraints set by `lb` and `ub` in the `OptimizationProblem` are required.

## Examples 

The Rosenbrock function can optimized using `MultistartOptimization.TikTak()` with 100 initial points and the local method `NLopt.LD_LBFGS()` as follows:

```julia
rosenbrock(x, p) =  (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p  = [1.0, 100.0]
f = OptimizationFunction(rosenbrock)
prob = Optimization.OptimizationProblem(f, x0, p, lb = [-1.0,-1.0], ub = [1.0,1.0])
sol = solve(prob, MultistartOptimization.TikTak(100), NLopt.LD_LBFGS())
```

You can use any `Optimization` optimizers you like. The global method of the `MultiStartOptimization` is a positional argument and followed by the local method. This for example means we can perform a multistartoptimization with LBFGS as the optimizer using either the `NLopt.jl` or `Optim.jl` implementation as follows. Moreover, this interface allows you access and adjust all the optimizer settings as you normally would:

```julia
rosenbrock(x, p) =  (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p  = [1.0, 100.0]
f = OptimizationFunction(rosenbrock)
prob = Optimization.OptimizationProblem(f, x0, p, lb = [-1.0,-1.0], ub = [1.0,1.0])
sol = solve(prob, MultistartOptimization.TikTak(100), NLopt.LD_LBFGS())
sol = solve(prob, MultistartOptimization.TikTak(100), LBFGS())
```
