# MultiStartOptimization.jl
[`MultistartOptimization`](https://github.com/tpapp/MultistartOptimization.jl) is a is a Julia package implementing a global optimization multistart method which performs local optimization after choosing multiple starting points.

`MultistartOptimization` requires both a global and local method to be defined. The global multistart method chooses a set of initial starting points from where local the local method starts from.

Currently, only one global method (`TikTak`) is implemented and called by `MultiStartOptimization.TikTak(n)` where `n` is the number of initial Sobol points. 

Currently, the local methods can be one of the algotithms implemented in `NLopt.jl`. 


!!! note

    If you checkout the master branch of `MultiStartOptimization` or have version `>=0.1.3` you can use all optimizers found in the `GalacticOptim` which work with an initial parameter set. See an example of this below.

## Global Optimizer
### Without Constraint Equations

The methods in [`MultistartOptimization`](https://github.com/tpapp/MultistartOptimization.jl) is performing global optimization on problems without
constraint equations. However, lower and upper constraints set by `lb` and `ub` in the `OptimizationProblem` are required.


The Rosenbrock function can optimized using `MultistartOptimization.TikTak()` with 100 initial points and the local method `NLopt.LD_LBFGS()` as follows:

```julia
rosenbrock(x, p) =  (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p  = [1.0, 100.0]
f = OptimizationFunction(rosenbrock)
prob = GalacticOptim.OptimizationProblem(f, x0, p, lb = [-1.0,-1.0], ub = [1.0,1.0])
sol = solve(prob, MultistartOptimization.TikTak(100), local_method = NLopt.LD_LBFGS())
```

If you have checked out the master branch `MultiStartOptimization` version `0.1.2` or have version `>=0.1.3` you can use any `GalactimOptim` optimizers you like. The global method of the `MultiStartOptimization` is a positional argument and if the given method has a local method it can be easily defined via the `local_method` keyword argument as you would without `MultiStartOptimization`. This for example means we can perform a multistartoptimization with LBFGS as the optimizer using either the `NLopt.jl` or `Optim.jl` implementation as follows:

```julia
rosenbrock(x, p) =  (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p  = [1.0, 100.0]
f = OptimizationFunction(rosenbrock)
prob = GalacticOptim.OptimizationProblem(f, x0, p, lb = [-1.0,-1.0], ub = [1.0,1.0])
sol = solve(prob, MultistartOptimization.TikTak(100), NLopt.LD_LBFGS())
sol = solve(prob, MultistartOptimization.TikTak(100), LBFGS())
```