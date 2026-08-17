# Optimization Interface

Optimization.jl provides the user-facing [`solve`](@ref) interface and a
cache-based interface for solver packages. Users normally construct an
`OptimizationProblem`, load a solver package, and call `solve`. Solver authors
use `init` and `solve!` when they need to inspect or continue a run.

## User Interface

```julia
using Optimization, OptimizationOptimJL

f(u, p) = sum(abs2, u)
prob = OptimizationProblem(OptimizationFunction(f), [1.0, -2.0])
sol = solve(prob, Optim.BFGS())
```

The problem and algorithm are validated through the capability traits in
`SciMLBase`. A solver that does not support bounds, constraints,
callbacks, gradients, Hessians, or other features should declare that through
the corresponding trait methods so incompatible problems fail before the
solver is called.

## Solver Extension Interface

A solver package can implement the cache interface by defining a concrete
cache type and methods for the SciMLBase extension points:

```julia
SciMLBase.__init(prob::OptimizationProblem, alg; kwargs...) -> cache
SciMLBase.__solve(cache) -> solution
```

The first dispatch argument should be a problem or cache type owned by the
solver package. This keeps the extension discoverable and avoids broad methods
that interfere with other solver packages. The cache should subtype
`SciMLBase.AbstractOptimizationCache`, and the solution should satisfy
`SciMLBase.AbstractOptimizationSolution`.

The cache is a developer-facing object. Solver packages may expose additional
documented methods for reinitialization, stopping, or inspecting progress, but
users should not depend on undocumented concrete cache fields. A solver that
does not support initialization can instead implement
`SciMLBase.__solve(prob, alg; kwargs...)` directly.
