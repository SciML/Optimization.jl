# Optimization Problem Reusage and Caching Interface

One of the key features for high-performance optimization workflows is the ability to reuse and modify existing optimization problems without full reconstruction. Optimization.jl provides several mechanisms for efficiently reusing optimization setups, particularly through the `reinit!` function and caching interfaces.

## Basic Reusage with `reinit!`

The `reinit!` function allows you to modify an existing optimization problem and reuse the solver setup. This is particularly useful for parameter sweeps, sensitivity analysis, and warm-starting optimization problems.

```julia
using Optimization, OptimizationOptimJL

# Define the Rosenbrock function
rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2

# Initial setup
x0 = [0.0, 0.0]
p = [1.0, 100.0]
prob = OptimizationProblem(rosenbrock, x0, p)

# Solve the first optimization
sol1 = solve(prob, BFGS())

# Reinitialize with new parameters without reconstructing the entire problem
reinit!(prob, x0, p = [2.0, 100.0])
sol2 = solve(prob, BFGS())

# Reinitialize with new initial conditions
reinit!(prob, [1.0, 1.0], p = [1.0, 100.0])
sol3 = solve(prob, BFGS())
```

## Parameter Sweeps

The `reinit!` function is particularly powerful for parameter sweeps where you need to solve the same optimization problem with different parameter values:

```julia
using Optimization, OptimizationOptimJL

# Define optimization problem
f(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = [0.0, 0.0]
prob = OptimizationProblem(f, x0, [1.0, 100.0])

# Parameter sweep
parameter_values = [50.0, 100.0, 150.0, 200.0]
solutions = []

for p_val in parameter_values
    reinit!(prob, x0, p = [1.0, p_val])
    sol = solve(prob, BFGS())
    push!(solutions, sol)
end

# Access results
for (i, sol) in enumerate(solutions)
    println("Parameter $(parameter_values[i]): Minimum at $(sol.u)")
end
```

## Warm-Starting Optimization

You can use previous solutions as starting points for new optimizations, which can significantly improve convergence:

```julia
using Optimization, OptimizationOptimJL

# Define a more complex optimization problem
complex_objective(x, p) = sum((x[i] - p[i])^2 for i in 1:length(x)) + 
                         sum(sin(x[i]) for i in 1:length(x))

# Initial problem
n = 10
x0 = zeros(n)
p0 = ones(n)
prob = OptimizationProblem(complex_objective, x0, p0)

# Solve initial problem
sol1 = solve(prob, BFGS())

# Use previous solution as warm start for new parameter set
new_params = 1.1 * ones(n)  # Slightly different parameters
reinit!(prob, sol1.u, p = new_params)  # Warm start with previous solution
sol2 = solve(prob, BFGS())

# Compare convergence
println("Initial problem converged in $(sol1.iterations) iterations")
println("Warm-started problem converged in $(sol2.iterations) iterations")
```

## Advanced Caching with Iterator Interface

For more advanced use cases, you can use the solver's iterator interface to have fine-grained control over the optimization process:

```julia
using Optimization, OptimizationOptimJL

# Setup optimization problem
rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
prob = OptimizationProblem(rosenbrock, [0.0, 0.0], [1.0, 100.0])

# Initialize the solver (this creates the cache)
cache = init(prob, BFGS())

# Perform optimization steps manually
for i in 1:10
    step!(cache)
    println("Step $i: Current point = $(cache.u), Objective = $(cache.objective)")
end

# Get final solution
sol = solve!(cache)
```

## Performance Benefits

The reusage interface provides several performance advantages:

1. **Reduced Memory Allocation**: Reusing problem structures avoids repeated memory allocations
2. **Warm Starting**: Using previous solutions as initial guesses can reduce iterations needed
3. **Solver State Preservation**: Internal solver states (like Hessian approximations) can be preserved
4. **Batch Processing**: Efficient processing of multiple related optimization problems

## When to Use `reinit!` vs `remake`

- **Use `reinit!`** when:
  - You want to preserve solver internal state
  - Parameters or initial conditions change slightly
  - You're doing parameter sweeps or sensitivity analysis
  - Performance is critical

- **Use `remake`** when:
  - The problem structure changes significantly
  - You need a completely fresh start
  - Problem dimensions change

## Example: Complete Parameter Study

Here's a comprehensive example showing how to efficiently perform a parameter study:

```julia
using Optimization, OptimizationOptimJL, Plots

# Define objective function
objective(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2 + p[3] * x[1] * x[2]

# Setup base problem
x0 = [0.0, 0.0]
base_params = [1.0, 100.0, 0.0]
prob = OptimizationProblem(objective, x0, base_params)

# Parameter study over the third parameter
param3_range = -5.0:0.5:5.0
results = Dict()

for p3 in param3_range
    # Efficiently update only the changing parameter
    new_params = [1.0, 100.0, p3]
    reinit!(prob, x0, p = new_params)
    
    # Solve and store results
    sol = solve(prob, BFGS())
    results[p3] = (solution = sol.u, objective = sol.objective)
end

# Analyze results
objectives = [results[p3].objective for p3 in param3_range]
plot(param3_range, objectives, xlabel="Parameter 3", ylabel="Objective Value", 
     title="Parameter Study using reinit!")
```

This reusage interface makes Optimization.jl highly efficient for production optimization workflows where the same problem structure is solved repeatedly with different parameters or initial conditions.