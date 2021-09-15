# Common Solver Options

In GalacticOptim.jl, solving an `OptimizationProblem` is done via:

```julia
solve(prob,alg;kwargs...)
```

The arguments to `solve` are common across all of the optimizers.
These common arguments are:

- `maxiters` (the maximum number of iterations)
- `maxtime` (the maximum of time the optimization runs for)
- `abstol` (absolute tolerance in changes of the objective value)
- `reltol` (relative tolerance  in changes of the objective value)
- `cb` (a callback function)

If the chosen global optimzer employs a local optimization method a similiar set of common local optimizer arguments exists.
The common local optimizer arguments are:

- `local_method` (optimiser used for local optimization in global method)
- `local_maxiters` (the maximum number of iterations)
- `local_maxtime` (the maximum of time the optimization runs for)
- `local_abstol` (absolute tolerance in changes of the objective value)
- `local_reltol` (relative tolerance  in changes of the objective value)
- `local_options` (NamedTuple of keyword arguments for local optimizer)

Some optimizer algorithms have special keyword arguments documented in the
solver portion of the documentation and their respective documentation.
These arguments can be passed as `kwargs...` to `solve`. Similiarly, the special
kewyword arguments for the `local_method` of a global optimizer are passed as a
`NamedTuple` to `local_options`.

Over time we hope to cover more of these keyword arguments under the common interface.

If a common argument is not implemented for a optimizer a warning will be shown.  


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
