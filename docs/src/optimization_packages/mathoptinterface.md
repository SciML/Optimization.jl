# MathOptInterface.jl

[MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl) is a Julia
abstraction layer to interface with a variety of mathematical optimization solvers.

## Installation: OptimizationMOI.jl

To use this package, install the OptimizationMOI package:

```julia
import Pkg;
Pkg.add("OptimizationMOI");
```

## Details

As of now, the `Optimization` interface to `MathOptInterface` implements only
the `maxtime` common keyword argument.

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
  - The full list of optimizer options can be found in the [KNITRO Documentation](https://www.artelys.com/docs/knitro//3_referenceManual/callableLibraryAPI.html)

#### Juniper.jl (MathOptInterface)

  - [`Juniper.Optimizer`](https://github.com/lanl-ansi/Juniper.jl)
  - Juniper requires a nonlinear optimizer to be set via the `nl_solver` option,
    which must be a MathOptInterface-based optimizer. See the
    [Juniper documentation](https://github.com/lanl-ansi/Juniper.jl) for more
    detail.

```@example MOI
using Optimization, OptimizationMOI, Juniper, Ipopt
rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
_p = [1.0, 100.0]

f = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
prob = Optimization.OptimizationProblem(f, x0, _p)

opt = OptimizationMOI.MOI.OptimizerWithAttributes(Juniper.Optimizer,
                                                  "nl_solver" => OptimizationMOI.MOI.OptimizerWithAttributes(Ipopt.Optimizer,
                                                                                                             "print_level" => 0))
sol = solve(prob, opt)
```
