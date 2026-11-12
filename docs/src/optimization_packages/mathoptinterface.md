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

As of now, the `Optimization` interface to `MathOptInterface` implements only
the `maxtime` common keyword argument.

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

## Multipliers and warm starts

For MOI solvers, `get_dual(sol)` returns a snapshot with
`constraints` (in problem constraint order), `lower` and `upper` (variable-bound
multipliers), `reduced_costs` (the sum of bound multipliers), `dual_objective`,
and `constraint_primal`. An unavailable result is `nothing`; bound multipliers
are zero where no bound exists. Values use MOI's dual sign convention. Use
`get_dual(sol, i)` for the `i`th problem constraint, or
`get_dual(sol, OptimizationMOI.MOI.VariableIndex(i))` for a reduced cost.
For NLP-block solvers, `constraint_primal` is evaluated from the constraint
function at the returned primal point and is `nothing` if evaluation fails. For
affine/quadratic solvers, it comes from MOI's constraint primal results, including
the constraint expression's constant. Results marked as certificates are not
returned as point values or reusable multipliers. ModelingToolkit systems may
canonicalize constraint expressions, so their primal values and multiplier
signs follow the canonicalized rows rather than the original equation's left side.

For example, minimize `(x - p)^2` subject to `x ≤ 1`. After solving with `p = 2`,
reuse both the primal solution and dual multipliers to solve again with `p = 3`:

```@example IpoptWarmStart
using OptimizationBase, OptimizationMOI, Ipopt, ADTypes, ForwardDiff

cons!(res, x, p) = (res[1] = x[1]; nothing)
f = OptimizationFunction((x, p) -> (x[1] - p[1])^2, ADTypes.AutoForwardDiff(); cons = cons!)
prob = OptimizationProblem(f, [0.5], [2.0];
    lb = [0.0], ub = [3.0], lcons = [-Inf], ucons = [1.0])

cache = init(prob, Ipopt.Optimizer(); print_level = 0, warm_start_init_point = "yes")
sol = solve!(cache)
duals = get_dual(sol)
OptimizationMOI.reinit!(cache; u0 = sol.u, p = [3.0], dual_values = duals)
sol2 = solve!(cache)
(; x = round(sol2.u[1]; digits = 4), constraint_dual = round(get_dual(sol2, 1); digits = 4))
```

The second solution has `x ≈ 1` and a constraint multiplier of approximately
`-4` in MOI's sign convention.

For an affine or quadratic problem, the same getters work with HiGHS:

```@example HiGHSWarmStart
using OptimizationBase, OptimizationMOI, HiGHS, ADTypes, ModelingToolkitBase

cons!(res, x, p) = (res[1] = x[1] + 2; nothing)
f = OptimizationFunction((x, p) -> (x[1] - 2)^2 + 7, ADTypes.AutoSymbolics(); cons = cons!)
prob = OptimizationProblem(f, [0.5]; lb = [0.0], ub = [3.0],
    lcons = [-Inf], ucons = [3.0])
cache = init(prob, HiGHS.Optimizer(); output_flag = false)
sol = solve!(cache)
duals = get_dual(sol)
OptimizationMOI.reinit!(cache; u0 = sol.u) # HiGHS supports primal starts
sol2 = solve!(cache)
(; x = round(sol2.u[1]; digits = 4), constraint_primal = round(duals.constraint_primal[1]; digits = 4),
    constraint_dual = round(duals.constraints[1]; digits = 4))
```

`dual_objective` is solver-reported; for quadratic problems it need not equal
the primal objective.

For NLP problems, changing `p` with `reinit!` rebuilds parameter-bound derivative
callbacks. If the rebuilt callback or derivative-buffer types change, create a new
cache instead. Pass `p` explicitly even after mutating a parameter object in place.
For affine/quadratic problems, the expressions are re-evaluated when the model is
rebuilt. A previous solution's dual results remain snapshots, but its cache still
shares the current parameter object.

Alternatively, pass `dual_values = (constraints = [...], lower = [...], upper = [...])`
to `init`, `solve`, or `reinit!`, omitting any starts not needed. Constraint and
bound dual starts require support from the optimizer. `lower_primal` and
`upper_primal` can supply bound constraint primal starts when supported; MOI's
NLP block has no constraint-primal-start attribute. On the affine/quadratic path,
`constraint_primal_start` can supply starts in constraint order. Entries for absent
variable bounds are ignored. Optimizers may require their own options to actually
use starts, as Ipopt does above. HiGHS supports variable primal starts, but not
MOI constraint primal or dual starts; explicitly requesting unsupported starts
raises an error. For a HiGHS warm start, use `reinit!(cache; u0 = sol.u)` without
passing the dual tuple.
`get_dual` snapshots remain valid after reuse, but `sol.original` may be emptied
when an optimizer instance is reused. Ipopt may not compute fixed-variable bound
multipliers with its default `fixed_variable_treatment`; consult its options when
those multipliers are needed.

```@docs
OptimizationMOI.get_dual
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
