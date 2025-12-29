# [OptimizationLBFGSB.jl](@id lbfgsb)

[`OptimizationLBFGSB.jl`](https://github.com/SciML/Optimization.jl/tree/master/lib/OptimizationLBFGSB) is a package that wraps the [L-BFGS-B](https://users.iems.northwestern.edu/%7Enocedal/lbfgsb.html) fortran routine via the [LBFGSB.jl](https://github.com/Gnimuc/LBFGSB.jl/) package.

## Installation

To use this package, install the `OptimizationLBFGSB` package:

```julia
using Pkg
Pkg.add("OptimizationLBFGSB")
```

## Methods

  - `LBFGSB`: The popular quasi-Newton method that leverages limited memory BFGS approximation of the inverse of the Hessian. It directly supports box-constraints.

    This can also handle arbitrary non-linear constraints through an Augmented Lagrangian method with bounds constraints described in 17.4 of Numerical Optimization by Nocedal and Wright. Thus serving as a general-purpose nonlinear optimization solver.

```@docs
OptimizationLBFGSB.LBFGSB
```

## Examples

### Unconstrained rosenbrock problem

```@example LBFGSB
using OptimizationBase, OptimizationLBFGSB, ADTypes, Zygote

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p = [1.0, 100.0]

optf = OptimizationFunction(rosenbrock, ADTypes.AutoZygote())
prob = OptimizationProblem(optf, x0, p)
sol = solve(prob, LBFGSB())
```

### With nonlinear and bounds constraints

```@example LBFGSB
function con2_c(res, x, p)
    res .= [x[1]^2 + x[2]^2, (x[2] * sin(x[1]) + x[1]) - 5]
end

optf = OptimizationFunction(rosenbrock, ADTypes.AutoZygote(), cons = con2_c)
prob = OptimizationProblem(optf, x0, p, lcons = [1.0, -Inf],
    ucons = [1.0, 0.0], lb = [-1.0, -1.0],
    ub = [1.0, 1.0])
res = solve(prob, LBFGSB(), maxiters = 100)
```
