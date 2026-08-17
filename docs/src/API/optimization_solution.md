# [Optimization Solutions](@id solution)

Optimization solver packages return `SciMLBase.OptimizationSolution`, which is
defined and documented by
[SciMLBase](https://github.com/SciML/SciMLBase.jl).
The solution stores the minimizer, objective value, return code, solver
statistics, and (for conic problems) optional dual multipliers. The
`calculate_dual` option controls whether dual multipliers are computed.
