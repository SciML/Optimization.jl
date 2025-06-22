# ModelingToolkit Integration

Optimization.jl is heavily integrated with the ModelingToolkit.jl
symbolic system for symbolic-numeric optimizations. It provides a
front-end for automating the construction, parallelization, and
optimization of code. Optimizers can better interface with the extra
symbolic information provided by the system.

There are two ways that the user interacts with ModelingToolkit.jl.
One can use `OptimizationFunction` with `AutoModelingToolkit` for
automatically transforming numerical codes into symbolic codes. See
the [OptimizationFunction documentation](@ref optfunction) for more
details.

Secondly, one can generate `OptimizationProblem`s for use in
Optimization.jl from purely a symbolic front-end. This is the form
users will encounter when using ModelingToolkit.jl directly, and it is
also the form supplied by domain-specific languages. For more information,
see the [OptimizationSystem documentation](https://docs.sciml.ai/ModelingToolkit/stable/API/problems/#SciMLBase.OptimizationProblem).
