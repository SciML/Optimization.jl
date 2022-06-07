# MathOptInterface.jl

[MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl) is Julia
abstration layer to interface with variety of mathematical optimization solvers.

## Installation: OptimizationMOI.jl

To use this package, install the OptimizationMOI package:

```julia
import Pkg; Pkg.add("OptimizationMOI")
```

## Details

As of now, the `Optimization` interface to `MathOptInterface` implements only
the `maxtime` common keyword argument. 

An optimizer which supports the `MathOptInterface` API can be called be called
directly if no optimizer options have to be defined. 
    
For example using the [`Ipopt.jl`](https://github.com/jump-dev/Ipopt.jl)
optimizer:


```julia
sol = solve(prob, Ipopt.Optimizer())
```

The optimizer options are handled in one of two ways. They can either be set via
`Optimization.MOI.OptimizerWithAttributes()` or as keyword argument to `solve`. 

For example using the `Ipopt.jl` optimizer:

```julia
opt = Optimization.MOI.OptimizerWithAttributes(Ipopt.Optimizer, "option_name" => option_value, ...)
sol = solve(prob, opt)

sol = solve(prob,  Ipopt.Optimizer(); option_name = option_value, ...)
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

```julia
using Optimization, ForwardDiff
rosenbrock(x, p) =  (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
_p  = [1.0, 100.0]

f = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
prob = Optimization.OptimizationProblem(f, x0, _p)

using Juniper, Ipopt
opt = Optimization.MOI.OptimizerWithAttributes(
    Juniper.Optimizer,
    "nl_solver"=>Optimization.MOI.OptimizerWithAttributes(Ipopt.Optimizer, "print_level"=>0),
)
sol = solve(prob, opt)
```
