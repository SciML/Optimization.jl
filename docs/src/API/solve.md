# Common Solver Options

In GalacticOptim.jl, solving an `OptimizationProblem` is done via:

```julia
solve(prob,alg;kwargs...)
```

The arguments to `solve` are common across all of the optimizers.
These common arguments are:

- `maxiters` (the maximum number of iterations)
- `abstol` (absolute tolerance)
- `reltol` (relative tolerance)
- `cb` (a callback function)

Some optimizer algorithms have special keyword arguments documented in the
solver portion of the documentation. Over time we hope to cover more of these
keyword arguments under the common interface.

## Callback Functions

The callback function `cb` is a function which is called after every optimizer
step. Its signature is:

```julia
cb = (x,other_args) -> nothing
```

where `other_args` is are the extra return arguments of the optimization `f`.
For example, if `f(x,p) = 5x`, then `cb = (x) -> ...` is used. If `f(x,p) = 5x,55x`,
then `cb = (x,extra) -> ...` is used, where `extra = 55x`. This allows for saving
values from the optimization and using them for plotting and display without
recalculating.
