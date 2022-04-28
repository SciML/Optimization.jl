# MathOptInterface.jl

[MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl) is Julia abstration layer to interface with variety of mathematical optimization solvers.

## Installation: GalacticMOI.jl

To use this package, install the GalacticMOI package:

```julia
import Pkg; Pkg.add("GalacticMOI")
```

## Details

As of now the `GalacticOptim` interface to `MathOptInterface` implents only the `maxtime` common keyword arguments. An optimizer which is implemented in the `MathOptInterface` is can be called be called directly if no optimizer options have to be defined. For example using the `Ipopt.jl` optimizer:

```julia
sol = solve(prob, Ipopt.Optimizer())
```

The optimizer options are handled in one of two ways. They can either be set via `GalacticOptim.MOI.OptimizerWithAttributes()` or as keyword argument to `solve`. For example using the `Ipopt.jl` optimizer:

```julia
opt = GalacticOptim.MOI.OptimizerWithAttributes(Ipopt.Optimizer, "option_name" => option_value, ...)
sol = solve(prob, opt)

sol = solve(prob,  Ipopt.Optimizer(); option_name = option_value, ...)
```



## Local Optimizer

### Local constraint
#### Ipopt.jl (MathOptInterface)

- [`Ipopt.Optimizer`](https://juliahub.com/docs/Ipopt/yMQMo/0.7.0/)
- Ipopt is a MathOptInterface optimizer, and thus its options are handled via
  `GalacticOptim.MOI.OptimizerWithAttributes(Ipopt.Optimizer, "option_name" => option_value, ...)`
- The full list of optimizer options can be found in the [Ipopt Documentation](https://coin-or.github.io/Ipopt/OPTIONS.html#OPTIONS_REF)

#### KNITRO.jl (MathOptInterface)

- [`KNITRO.Optimizer`](https://github.com/jump-dev/KNITRO.jl)
- KNITRO is a MathOptInterface optimizer, and thus its options are handled via
  `GalacticOptim.MOI.OptimizerWithAttributes(KNITRO.Optimizer, "option_name" => option_value, ...)`
- The full list of optimizer options can be found in the [KNITRO Documentation](https://www.artelys.com/docs/knitro//3_referenceManual/callableLibraryAPI.html)

#### AmplNLWriter.jl (MathOptInterface)

- [`AmplNLWriter.Optimizer`](https://github.com/jump-dev/AmplNLWriter.jl)
- AmplNLWriter is a MathOptInterface optimizer, and thus its options are handled via
  `GalacticOptim.MOI.OptimizerWithAttributes(AmplNLWriter.Optimizer(algname), "option_name" => option_value, ...)`
- Possible `algname`s are:
    * `Bonmin_jll.amplexe`
    * `Couenne_jll.amplexe`
    * `Ipopt_jll.amplexe`
    * `SHOT_jll.amplexe`

To use one of the JLLs, they must be added first. For example: `Pkg.add("Bonmin_jll")`.

#### Juniper.jl (MathOptInterface)

- [`Juniper.Optimizer`](https://github.com/lanl-ansi/Juniper.jl)
- Juniper is a MathOptInterface optimizer, and thus its options are handled via
  `GalacticOptim.MOI.OptimizerWithAttributes(Ipopt.Optimizer, "option_name" => option_value, ...)`
- Juniper requires the choice of a relaxation method `nl_solver` which must be
  a MathOptInterface-based optimizer

```julia
using GalacticOptim, ForwardDiff
rosenbrock(x, p) =  (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
_p  = [1.0, 100.0]

f = OptimizationFunction(rosenbrock, GalacticOptim.AutoForwardDiff())
prob = GalacticOptim.OptimizationProblem(f, x0, _p)

using Juniper, Ipopt
optimizer = Juniper.Optimizer
# Choose a relaxation method
nl_solver = GalacticOptim.MOI.OptimizerWithAttributes(Ipopt.Optimizer, "print_level"=>0)

opt = GalacticOptim.MOI.OptimizerWithAttributes(optimizer, "nl_solver"=>nl_solver)
sol = solve(prob, opt)
```


## BARON.jl (MathOptInterface)

- [`BARON.Optimizer`](https://github.com/joehuchette/BARON.jl)
- BARON is a MathOptInterface optimizer, and thus its options are handled via
  `GalacticOptim.MOI.OptimizerWithAttributes(BARON.Optimizer, "option_name" => option_value, ...)`
- The full list of optimizer options can be found in the [BARON Documentation](https://minlp.com/baron-solver)


### Gradient-Based
#### Ipopt.jl (MathOptInterface)

- [`Ipopt.Optimizer`](https://juliahub.com/docs/Ipopt/yMQMo/0.7.0/)
- Ipopt is a MathOptInterface optimizer, and thus its options are handled via
  `GalacticOptim.MOI.OptimizerWithAttributes(Ipopt.Optimizer, "option_name" => option_value, ...)`
- The full list of optimizer options can be found in the [Ipopt Documentation](https://coin-or.github.io/Ipopt/OPTIONS.html#OPTIONS_REF)


## Global Optimizer

### With Constraint Equations
#### Alpine.jl (MathOptInterface)

- [`Alpine.Optimizer`](https://github.com/lanl-ansi/Alpine.jl)
- Alpine is a MathOptInterface optimizer, and thus its options are handled via
  `GalacticOptim.MOI.OptimizerWithAttributes(Alpine.Optimizer, "option_name" => option_value, ...)`
- The full list of optimizer options can be found in the [Alpine Documentation](https://github.com/lanl-ansi/Alpine.jl)
