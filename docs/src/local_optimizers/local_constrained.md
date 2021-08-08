# Local Constrained Optimization

Local constrained optimization methods are fast methods for finding a
local optima which satisfies the desired constraints.

## Recommended Methods

`Ipopt` is recommended if standard `Float64` and `Array` values are used (these
are requirements because `Ipopt` is a C library). Otherwise `Optim.IPNewton`
is recommended.

## Optim.jl

- [`Optim.IPNewton`](https://julianlsolvers.github.io/Optim.jl/stable/#algo/ipnewton/)
    * `linesearch` specifies the line search algorithm (for more information, consult [this source](https://github.com/JuliaNLSolvers/LineSearches.jl) and [this example](https://julianlsolvers.github.io/LineSearches.jl/latest/examples/generated/optim_linesearch.html))
        * available line search algorithms:
        * `HaegerZhang`
        * `MoreThuente`
        * `BackTracking`
        * `StrongWolfe`
        * `Static`
    * `μ0` specifies the initial barrier penalty coefficient as either a number or `:auto`
    * `show_linesearch` is an option to turn on linesearch verbosity.
    * Defaults:
        * `linesearch::Function = Optim.backtrack_constrained_grad`
        * `μ0::Union{Symbol,Number} = :auto`
        * `show_linesearch::Bool = false`

### Optim Keyword Arguments

The following special keyword arguments can be used with Optim.jl optimizers:

* `x_tol`: Absolute tolerance in changes of the input vector `x`, in infinity norm. Defaults to `0.0`.
* `f_tol`: Relative tolerance in changes of the objective value. Defaults to `0.0`.
* `g_tol`: Absolute tolerance in the gradient, in infinity norm. Defaults to `1e-8`. For gradient free methods, this will control the main convergence tolerance, which is solver specific.
* `f_calls_limit`: A soft upper limit on the number of objective calls. Defaults to `0` (unlimited).
* `g_calls_limit`: A soft upper limit on the number of gradient calls. Defaults to `0` (unlimited).
* `h_calls_limit`: A soft upper limit on the number of Hessian calls. Defaults to `0` (unlimited).
* `allow_f_increases`: Allow steps that increase the objective value. Defaults to `false`. Note that, when setting this to `true`, the last iterate will be returned as the minimizer even if the objective increased.
* `iterations`: How many iterations will run before the algorithm gives up? Defaults to `1_000`.
* `store_trace`: Should a trace of the optimization algorithm's state be stored? Defaults to `false`.
* `show_trace`: Should a trace of the optimization algorithm's state be shown on `stdout`? Defaults to `false`.
* `extended_trace`: Save additional information. Solver dependent. Defaults to `false`.
* `trace_simplex`: Include the full simplex in the trace for `NelderMead`. Defaults to `false`.
* `show_every`: Trace output is printed every `show_every`th iteration.
* `time_limit`: A soft upper limit on the total run time. Defaults to `NaN` (unlimited).

## Ipopt.jl (MathOptInterface)

- [`Ipopt.Optimizer`](https://juliahub.com/docs/Ipopt/yMQMo/0.7.0/)
- Ipopt is a MathOptInterface optimizer, and thus its options are handled via
  `GalacticOptim.MOI.OptimizerWithAttributes(Ipopt.Optimizer, "option_name" => option_value, ...)`
- The full list of optimizer options can be found in the [Ipopt Documentation](https://coin-or.github.io/Ipopt/OPTIONS.html#OPTIONS_REF)

## KNITRO.jl (MathOptInterface)

- [`KNITRO.Optimizer`](https://github.com/jump-dev/KNITRO.jl)
- KNITRO is a MathOptInterface optimizer, and thus its options are handled via
  `GalacticOptim.MOI.OptimizerWithAttributes(KNITRO.Optimizer, "option_name" => option_value, ...)`
- The full list of optimizer options can be found in the [KNITRO Documentation](https://www.artelys.com/docs/knitro//3_referenceManual/callableLibraryAPI.html)

## AmplNLWriter.jl (MathOptInterface)

- [`AmplNLWriter.Optimizer`](https://github.com/jump-dev/AmplNLWriter.jl)
- AmplNLWriter is a MathOptInterface optimizer, and thus its options are handled via
  `GalacticOptim.MOI.OptimizerWithAttributes(AmplNLWriter.Optimizer(algname), "option_name" => option_value, ...)`
- Possible `algname`s are:
    * `Bonmin_jll.amplexe`
    * `Couenne_jll.amplexe`
    * `Ipopt_jll.amplexe`
    * `SHOT_jll.amplexe`

To use one of the JLLs, they must be added first. For example: `Pkg.add("Bonmin_jll")`.

## Juniper.jl (MathOptInterface)

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
prob = OptimizationProblem(f, x0, _p)

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

## NLopt.jl

NLopt.jl algorithms are chosen via `NLopt.Opt(:algname)`. Consult the
[NLopt Documentation](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/)
for more information on the algorithms. Possible algorithm names are:

* `:LN_COBYLA`
* `:LD_CCSAQ`
* `:LD_SLSQP`
* `:LD_LBFGS`
* `:LD_TNEWTON_PRECOND_RESTART`
* `:LD_TNEWTON_PRECOND`
* `:LD_TNEWTON_RESTART`
* `:LD_TNEWTON`
* `:LD_VAR2`
* `:LD_VAR1`
* `:AUGLAG`
* `:AUGLAG_EQ`
