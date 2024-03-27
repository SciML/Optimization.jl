# PRIMA.jl

[PRIMA.jl](https://github.com/libprima/PRIMA.jl) is a julia wrapper for the fortran library [prima](https://github.com/libprima/prima) which implements Powell's derivative free optimization methods.

## Installation: OptimizationPRIMA

To use this package, install the OptimizationPRIMA package:

```julia
import Pkg;
Pkg.add("OptimizationPRIMA");
```

## Local Optimizer

The five Powell's algorithms of the prima library are provided by the PRIMA.jl package:

`UOBYQA`: (Unconstrained Optimization BY Quadratic Approximations) is for unconstrained optimization, that is Ω = ℝⁿ.

`NEWUOA`: is also for unconstrained optimization. According to M.J.D. Powell, newuoa is superior to uobyqa.

`BOBYQA`: (Bounded Optimization BY Quadratic Approximations) is for simple bound constrained problems, that is Ω = { x ∈ ℝⁿ | xl ≤ x ≤ xu }.

`LINCOA`: (LINearly Constrained Optimization) is for constrained optimization problems with bound constraints, linear equality constraints, and linear inequality constraints.

`COBYLA`: (Constrained Optimization BY Linear Approximations) is for general constrained problems with bound constraints, non-linear constraints, linear equality constraints, and linear inequality constraints.

```@example PRIMA
using Optimization, OptimizationPRIMA

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
_p = [1.0, 100.0]

prob = OptimizationProblem(rosenbrock, x0, _p)

sol = Optimization.solve(prob, UOBYQA(), maxiters = 1000)

sol = Optimization.solve(prob, NEWUOA(), maxiters = 1000)

sol = Optimization.solve(prob, BOBYQA(), maxiters = 1000)

sol = Optimization.solve(prob, LINCOA(), maxiters = 1000)

function con2_c(res, x, p)
    res .= [x[1] + x[2], x[2] * sin(x[1]) - x[1]]
end
optprob = OptimizationFunction(rosenbrock, cons = con2_c)
prob = OptimizationProblem(optprob, x0, _p, lcons = [1, -100], ucons = [1, 100])
sol = Optimization.solve(prob, COBYLA(), maxiters = 1000)
```
