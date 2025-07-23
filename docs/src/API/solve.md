# Common Solver Options (Solve Keyword Arguments)

```@docs
solve(::OptimizationProblem,::Any)
```

## Understanding the Solution Object

The `solve` function returns an `OptimizationSolution` object that contains the results of the optimization. This object has several important fields that provide information about the solution and the optimization process.

### Main Solution Fields

- **`sol.u`** or **`sol.minimizer`**: The optimal values found by the optimizer. These are aliases - both refer to the same solution vector. Use whichever feels more natural for your problem context.
  
- **`sol.objective`** or **`sol.minimum`**: The objective function value at the solution point. These are also aliases for the same value.

- **`sol.retcode`**: The return code indicating how the optimization terminated. Common values include:
  - `ReturnCode.Success`: Optimization converged successfully
  - `ReturnCode.MaxIters`: Maximum iterations reached
  - `ReturnCode.MaxTime`: Maximum time limit reached
  - `ReturnCode.Failure`: Optimization failed (check solver output for details)

- **`sol.stats`**: Statistics about the optimization process, including:
  - `iterations`: Number of iterations performed
  - `time`: Total computation time
  - `fevals`: Number of objective function evaluations
  - `gevals`: Number of gradient evaluations (if applicable)

- **`sol.original`**: The original output from the underlying solver package (solver-specific)

- **`sol.cache`**: The optimization cache used during solving (advanced use)

### Example Usage

```julia
using Optimization, OptimizationOptimJL

# Define and solve an optimization problem
rosenbrock(u, p) = (p[1] - u[1])^2 + p[2] * (u[2] - u[1]^2)^2
optf = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, zeros(2), [1.0, 100.0])
sol = solve(prob, Optim.BFGS())

# Access solution information
println("Solution: ", sol.u)  # or sol.minimizer
println("Objective value: ", sol.objective)  # or sol.minimum
println("Return code: ", sol.retcode)
println("Iterations: ", sol.stats.iterations)
println("Function evaluations: ", sol.stats.fevals)
println("Total time: ", sol.stats.time, " seconds")

# Check if optimization was successful
if sol.retcode == ReturnCode.Success
    println("Optimization converged successfully!")
end
```

### Array Interface

The solution object also supports an array interface for convenience:

```julia
# Access solution components like an array
sol[1]  # First component of the solution vector
sol[:]  # All components (same as sol.u)
length(sol)  # Number of optimization variables
```
