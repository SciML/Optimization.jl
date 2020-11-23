# Usage

## Automatic Differentiation Choices

While one can fully define all of the derivative functions associated with
nonlinear constrained optimization directly, in many cases it's easiest to just
rely on automatic differentiation to derive those functions. In GalacticOptim.jl,
you can provide as few functions as you want, or give a differentiation library
choice.

- `AutoForwardDiff()`
- `AutoReverseDiff(compile=false)`
- `AutoTracker()`
- `AutoZygote()`
- `AutoFiniteDiff()`
- `AutoModelingToolkit()`


## API Documentation

```julia
OptimizationFunction(f, AutoForwardDiff();
                     grad = nothing,
                     hes = nothing,
                     hv = nothing,
                     chunksize = 1)
```

```julia
OptimizationProblem(f, x, p = DiffEqBase.NullParameters(),;
                    lb = nothing,
                    ub = nothing)
```

```julia
solve(prob,alg;kwargs...)
```

Keyword arguments:

  - `maxiters` (the maximum number of iterations)
  - `abstol` (absolute tolerance)
  - `reltol` (relative tolerance)
