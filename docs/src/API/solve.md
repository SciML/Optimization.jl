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
