# [Optimization Solutions](@id solution)

```@docs
SciMLBase.OptimizationSolution
```

## Accessing Solution Information

The `OptimizationSolution` type is returned by the `solve` function and contains all information about the optimization result. Here's a quick reference for the most commonly used fields:

### Solution Values
- `sol.u` or `sol.minimizer`: The optimal parameter values
- `sol.objective` or `sol.minimum`: The objective function value at the optimum

### Convergence Information
- `sol.retcode`: Return code indicating termination reason
- `sol.stats`: Detailed statistics about the optimization process

### Aliases
For user convenience, the following aliases are provided:
- `sol.minimizer` is an alias for `sol.u`
- `sol.minimum` is an alias for `sol.objective`

These aliases allow you to use terminology that feels more natural for your specific problem context.

For detailed examples and usage patterns, see the [Common Solver Options](@ref) section.
