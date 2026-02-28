# SimpleOptimization.jl

SimpleOptimization.jl provides lightweight, loop-unrolled optimization algorithms for the SciML ecosystem. It follows the same design principle as other "Simple" packages in the SciML organization:

- **DifferentialEquations.jl** ↔ **SimpleDiffEq.jl**
- **NonlinearSolve.jl** ↔ **SimpleNonlinearSolve.jl**
- **Optimization.jl** ↔ **SimpleOptimization.jl**

While Optimization.jl captures all cases with extensive features and solver options, SimpleOptimization.jl provides much simpler code for just the basic cases. This means:

- Faster precompilation times
- Support for static arrays
- Compiles to GPU kernels

## Algorithms

- `SimpleGradientDescent(; eta=0.01)` - Lightweight gradient descent optimizer
- `SimpleBFGS()` - Lightweight BFGS quasi-Newton optimizer
- `SimpleLBFGS(; threshold=Val(10))` - Lightweight Limited-memory BFGS optimizer
- `SimpleNewton()` - Lightweight Newton optimizer (uses Hessian via nested AD)

## Documentation

See the [Optimization.jl documentation](https://docs.sciml.ai/Optimization/stable/optimization_packages/simpleoptimization/) for usage details.
