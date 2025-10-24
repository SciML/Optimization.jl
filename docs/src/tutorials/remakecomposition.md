# Creating polyalgorithms by chaining solvers using `remake`

The general framework of using multiple solvers to use exploration-convergence alternations is commonly
known as polyalgorithms. In the past Optimization.jl has provided a `PolyOpt` solver in [`OptimizationPolyalgorithms.jl`](@ref) which combined Adam from Optimisers.jl with BFGS from Optim.jl.
With the large number of choices available through the interface unique combinations of solvers can be effective for specific problems.

In this tutorial we will demonstrate how to use the `remake` function to chain together solvers to create your own polyalgorithms.

The SciML interface provides a `remake` function which allows you to recreate the `OptimizationProblem` from a previously defined `OptimizationProblem` with different initial guess for the optimization variables.

Let's look at a 10 dimensional schwefel function in the hypercube $x_i \in [-500, 500]$.

```@example polyalg
using OptimizationLBFGSB, Random
using OptimizationBBO, ReverseDiff

Random.seed!(122333)

function f_schwefel(x, p = [418.9829])
    result = p[1] * length(x)
    for i in 1:length(x)
        result -= x[i] * sin(sqrt(abs(x[i])))
    end
    return result
end

optf = OptimizationFunction(f_schwefel, Optimization.AutoReverseDiff(compile = true))

x0 = ones(10) .* 200.0
prob = OptimizationProblem(
    optf, x0, [418.9829], lb = fill(-500.0, 10), ub = fill(500.0, 10))

@show f_schwefel(x0)
```

Our polyalgorithm strategy will to use BlackBoxOptim's global optimizers for efficient exploration of the
parameter space followed by a quasi-Newton LBFGS method to (hopefully) converge to the global
optimum.

```@example polyalg
res1 = solve(prob, BBO_adaptive_de_rand_1_bin(), maxiters = 4000)

@show res1.objective
```

This is a good start can we converge to the global optimum?

```@example polyalg
prob = remake(prob, u0 = res1.minimizer)
res2 = solve(prob, LBFGS(), maxiters = 100)

@show res2.objective
```

Yay! We have found the global optimum (this is known to be at $x_i = 420.9687$).
