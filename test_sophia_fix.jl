using Optimization, ComponentArrays, Enzyme

# Test function - simple quadratic
function rosenbrock(x, p)
    return (1 - x.a)^2 + 100 * (x.b - x.a^2)^2
end

# Initial parameter as ComponentVector
x0 = ComponentVector(a = 0.0, b = 0.0)

# Create optimization function with Enzyme autodiff
optf = OptimizationFunction(rosenbrock, AutoEnzyme())

# Create optimization problem
prob = OptimizationProblem(optf, x0)

# Test that Sophia optimizer works without shadow generation errors
try
    sol = solve(prob, Optimization.Sophia(η=0.01, k=2), maxiters=5)
    println("✓ Sophia optimizer with ComponentArrays succeeded!")
    println("Solution: ", sol.u)
    println("Final objective: ", sol.objective)
catch e
    println("✗ Error: ", e)
    rethrow(e)
end