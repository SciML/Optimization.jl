# Optimization Problem Reusage and Caching Interface

## Reusing Optimization Caches with `reinit!`

The `reinit!` function allows you to efficiently reuse an existing optimization cache with new parameters or initial values. This is particularly useful when solving similar optimization problems repeatedly with different parameter values, as it avoids the overhead of creating a new cache from scratch.

### Basic Usage

```@example reinit
# Create initial problem and cache
using Optimization, OptimizationOptimJL
rosenbrock(u, p) = (p[1] - u[1])^2 + p[2] * (u[2] - u[1]^2)^2
u0 = zeros(2)
p = [1.0, 100.0]

optf = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, u0, p)

# Initialize cache and solve
cache = Optimization.init(prob, Optim.BFGS())
sol = Optimization.solve!(cache)

# Reinitialize cache with new parameters
cache = Optimization.reinit!(cache; p = [2.0, 50.0])
sol2 = Optimization.solve!(cache)
```

### Supported Arguments

The `reinit!` function supports updating various fields of the optimization cache:

  - `u0`: New initial values for the optimization variables
  - `p`: New parameter values
  - `lb`: New lower bounds (if applicable)
  - `ub`: New upper bounds (if applicable)
  - `lcons`: New lower bounds for constraints (if applicable)
  - `ucons`: New upper bounds for constraints (if applicable)

### Example: Parameter Sweep

```@example reinit
# Solve for multiple parameter values efficiently
results = []
p_values = [[1.0, 100.0], [2.0, 100.0], [3.0, 100.0]]

# Create initial cache
cache = Optimization.init(prob, Optim.BFGS())

function sweep(cache, p_values)
    for p in p_values
        cache = Optimization.reinit!(cache; p = p)
        sol = Optimization.solve!(cache)
        push!(results, (p = p, u = sol.u, objective = sol.objective))
    end
end

sweep(cache, p_values)
```

### Example: Updating Initial Values

```julia
# Warm-start optimization from different initial points
u0_values = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]

for u0 in u0_values
    local cache
    cache = Optimization.reinit!(cache; u0 = u0)
    sol = Optimization.solve!(cache)
    println("Starting from ", u0, " converged to ", sol.u)
end
```

### Performance Benefits

Using `reinit!` is more efficient than creating a new problem and cache for each parameter value, especially when:

  - The optimization algorithm maintains internal state that can be reused
  - The problem structure remains the same (only parameter values change)

### Notes

  - The `reinit!` function modifies the cache in-place and returns it for convenience
  - Not all fields need to be specified; only provide the ones you want to update
  - The function is particularly useful in iterative algorithms, parameter estimation, and when solving families of related optimization problems
  - For creating a new problem with different parameters (rather than modifying a cache), use `remake` on the `OptimizationProblem` instead
