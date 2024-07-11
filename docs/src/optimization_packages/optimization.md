# Optimization.jl

There are some solvers that are available in the Optimization.jl package directly without the need to install any of the solver wrappers.

## Methods

`LBFGS`: The popular quasi-Newton method that leverages limited memory BFGS approximation of the inverse of the Hessian. Through a wrapper over the [L-BFGS-B](https://users.iems.northwestern.edu/%7Enocedal/lbfgsb.html) fortran routine accessed from the [LBFGSB.jl](https://github.com/Gnimuc/LBFGSB.jl/) package. It directly supports box-constraints.

This can also handle arbitrary non-linear constraints through a Augmented Lagrangian method with bounds constraints described in 17.4 of Numerical Optimization by Nocedal and Wright. Thus serving as a general-purpose nonlinear optimization solver available directly in Optimization.jl.

## Examples

### Unconstrained rosenbrock problem

```@example L-BFGS
using Optimization, Zygote

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p = [1.0, 100.0]

optf = OptimizationFunction(rosenbrock, AutoZygote())
prob = Optimization.OptimizationProblem(optf, x0, p)
sol = solve(prob, Optimization.LBFGS())
```

### With nonlinear and bounds constraints

```@example L-BFGS
function con2_c(res, x, p)
    res .= [x[1]^2 + x[2]^2, (x[2] * sin(x[1]) + x[1]) - 5]
end

optf = OptimizationFunction(rosenbrock, AutoZygote(), cons = con2_c)
prob = OptimizationProblem(optf, x0, p, lcons = [1.0, -Inf],
    ucons = [1.0, 0.0], lb = [-1.0, -1.0],
    ub = [1.0, 1.0])
res = solve(prob, Optimization.LBFGS(), maxiters = 100)
```
