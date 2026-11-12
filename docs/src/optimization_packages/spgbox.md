# [SPGBox.jl](@id spgbox)

[`SPGBox.jl`](https://github.com/m3g/SPGBox.jl) is a pure-Julia implementation
of the Spectral Projected Gradient (SPG) method for box-constrained
optimization. It can be used through Optimization.jl with the `SPGBoxOpt`
algorithm.

SPG is a first-order, non-monotone method. Its line search uses the maximum of
the most recent `m` objective values as a reference, so an accepted step need
not decrease the objective relative to the immediately preceding iterate. The
method has very low memory requirements: it stores a fixed number of vectors
proportional to the number of decision variables, plus the length-`m` objective
history. It does not store or update a Hessian approximation.

## Installation

Install `SPGBox` (version 0.7.4 or later, which provides `SPGBoxOpt`) and
`Optimization`. An automatic differentiation package is also needed unless the
gradient is provided explicitly. The example below uses ForwardDiff:

```julia
using Pkg
Pkg.add(["SPGBox", "Optimization", "ForwardDiff"])
```

## Methods

`SPGBoxOpt()` minimizes differentiable objectives with optional lower and
upper bounds. The objective must have the Optimization.jl signature `f(x, p)`;
its gradient must be supplied through the `OptimizationFunction`, either by
an AD backend or an explicit `grad` function. SPGBox supports box bounds, but
not general nonlinear constraints.

The solver options accepted by `SPGBoxOpt` are:

- `eps`: stopping tolerance for the norm of the projected gradient (default
  `1e-5`).
- `nfevalmax`: maximum number of objective evaluations (default `1000`).
- `m`: number of recent objective values used in the non-monotone line search
  (default `10`).
- `project_x0`: whether to project the initial point onto the bounds (default
  `true`).
- `step_nc`: fallback spectral step length when the Barzilai-Borwein step
  cannot be computed (default `100`).
- `iprint`: print level (`nothing`, `0`, `1`, or `2`; default `nothing`).

The common Optimization.jl options `maxiters`, `maxtime`, and `callback` are
also supported. `maxiters` sets the maximum number of iterations; `maxtime` is
checked at the end of each iteration; and `callback` receives
`(state, objective)` and can stop the solve by returning `true`. `abstol` and
`reltol` are not used; set `eps` to control the projected-gradient tolerance.

## Example

```julia
using Optimization, SPGBox, ForwardDiff

objective(x, p) = (x[1] - p[1])^2 + (x[2] - 1)^2
optf = OptimizationFunction(objective, AutoForwardDiff())
prob = OptimizationProblem(
    optf,
    [10.0, 18.0],
    [1.0];
    lb = [2.0, -Inf],
    ub = [Inf, Inf],
)
sol = solve(prob, SPGBoxOpt())
```

The minimizer is `[2.0, 1.0]` with objective value `1.0`.

## References

E. G. Birgin, J. M. Martínez, and M. Raydan. “Nonmonotone spectral projected
gradient methods on convex sets.” *SIAM Journal on Optimization*, 10(4),
1196–1211, 2000. [https://doi.org/10.1137/S1052623497330963](https://doi.org/10.1137/S1052623497330963).
