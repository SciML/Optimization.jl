# [MathOptInterface.jl](@id mathoptinterface)

[MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl) is a Julia
abstraction layer to interface with a variety of mathematical optimization solvers.

## Installation: OptimizationMOI.jl

To use this package, install the OptimizationMOI package:

```julia
import Pkg;
Pkg.add("OptimizationMOI");
```

## Details

The `Optimization` interface to `MathOptInterface` implements the `maxtime` and
`maxiters` common keyword arguments. `maxiters` sets the solver's own iteration
limit, for the solvers listed below; other solvers ignore it with a message. A
limit set through the solver's own option, as a keyword argument or in
`OptimizerWithAttributes`, takes precedence over `maxiters`.

| Solver     | Option set by `maxiters`         |
|:---------- |:-------------------------------- |
| Ipopt      | `max_iter`                       |
| NLopt      | `maxeval`                        |
| Gurobi     | `IterationLimit`                 |
| CPLEX      | `CPX_PARAM_ITLIM`                |
| MOSEK      | `MSK_IPAR_INTPNT_MAX_ITERATIONS` |
| OSQP       | `max_iter`                       |
| ECOS       | `maxit`                          |
| SCS        | `max_iters`                      |
| COSMO      | `max_iter`                       |

The Gurobi and CPLEX options limit only simplex iterations, and the MOSEK option limits
only interior-point iterations. Other algorithms of those solvers, such as the barrier
method of Gurobi and CPLEX, are not limited by `maxiters`; set their limits through the
solver's own options.

`OptimizationMOI` supports an argument `mtkize` which takes a boolean (default to `false`)
that allows automatic symbolic expression generation, this allows using any AD backend with
solvers or interfaces such as AmplNLWriter that require the expression graph of the objective
and constraints. This always happens automatically in the case of the `AutoSymbolics`
`adtype`.

An optimizer which supports the `MathOptInterface` API can be called
directly if no optimizer options have to be defined.

For example, using the [`Ipopt.jl`](https://github.com/jump-dev/Ipopt.jl)
optimizer:

```julia
using OptimizationMOI, Ipopt
sol = solve(prob, Ipopt.Optimizer())
```

The optimizer options are handled in one of two ways. They can either be set via
`OptimizationMOI.MOI.OptimizerWithAttributes()` or as keyword arguments to `solve`.

For example, using the `Ipopt.jl` optimizer:

```julia
using OptimizationMOI, Ipopt
opt = OptimizationMOI.MOI.OptimizerWithAttributes(Ipopt.Optimizer,
    "option_name" => option_value, ...)
sol = solve(prob, opt)

sol = solve(prob, Ipopt.Optimizer(); option_name = option_value, ...)
```

## Optimizers

#### Ipopt.jl (MathOptInterface)

  - [`Ipopt.Optimizer`](https://github.com/jump-dev/Ipopt.jl)
  - The full list of optimizer options can be found in the [Ipopt Documentation](https://coin-or.github.io/Ipopt/OPTIONS.html#OPTIONS_REF)

#### KNITRO.jl (MathOptInterface)

  - [`KNITRO.Optimizer`](https://github.com/jump-dev/KNITRO.jl)
  - The full list of optimizer options can be found in the [KNITRO Documentation](https://www.artelys.com/app/docs/knitro/3_referenceManual/callableLibraryAPI.html)

#### Juniper.jl (MathOptInterface)

  - [`Juniper.Optimizer`](https://github.com/lanl-ansi/Juniper.jl)
  - Juniper requires a nonlinear optimizer to be set via the `nl_solver` option,
    which must be a MathOptInterface-based optimizer. See the
    [Juniper documentation](https://github.com/lanl-ansi/Juniper.jl) for more
    detail.

```@example MOI
using OptimizationBase, OptimizationMOI, Juniper, Ipopt, ADTypes, ForwardDiff
rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
_p = [1.0, 100.0]

f = OptimizationFunction(rosenbrock, ADTypes.AutoForwardDiff())
prob = OptimizationProblem(f, x0, _p)

opt = OptimizationMOI.MOI.OptimizerWithAttributes(
    Juniper.Optimizer,
    "nl_solver" => OptimizationMOI.MOI.OptimizerWithAttributes(
        Ipopt.Optimizer,
        "print_level" => 0
    )
)
sol = solve(prob, opt)
```

#### Using Integer Constraints

The following shows how to use integer linear programming within `Optimization`. We will solve the classical Knapsack Problem using `Juniper.jl`.

  - [`Juniper.Optimizer`](https://github.com/lanl-ansi/Juniper.jl)

  - Juniper requires a nonlinear optimizer to be set via the `nl_solver` option,
    which must be a MathOptInterface-based optimizer. See the
    [Juniper documentation](https://github.com/lanl-ansi/Juniper.jl) for more
    detail.
  - The integer domain is inferred based on the bounds of the variable:

      + Setting the lower bound to zero and the upper bound to one corresponds to `MOI.ZeroOne()` or a binary decision variable
      + Providing other or no bounds corresponds to `MOI.Integer()`

```@example MOI
v = [1.0, 2.0, 4.0, 3.0]
w = [5.0, 4.0, 3.0, 2.0]
W = 4.0
u0 = [0.0, 0.0, 0.0, 1.0]

optfun = OptimizationFunction(
    (u, p) -> -v'u, cons = (res, u, p) -> res .= w'u,
    ADTypes.AutoForwardDiff()
)

optprob = OptimizationProblem(
    optfun, u0; lb = zero.(u0), ub = one.(u0),
    int = ones(Bool, length(u0)),
    lcons = [-Inf;], ucons = [W;]
)

nl_solver = OptimizationMOI.MOI.OptimizerWithAttributes(
    Ipopt.Optimizer,
    "print_level" => 0
)
minlp_solver = OptimizationMOI.MOI.OptimizerWithAttributes(
    Juniper.Optimizer,
    "nl_solver" => nl_solver
)

res = solve(optprob, minlp_solver)
```
