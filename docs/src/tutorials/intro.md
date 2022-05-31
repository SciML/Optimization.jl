# Basic usage

In this tutorial we introduce the basics of GalcticOptim.jl by showing
how to easily mix local optimizers from Optim.jl and global optimizers
from BlackBoxOptim.jl on the Rosenbrock equation. The simplest copy-pasteable
code to get started is the following:

```julia
using Optimization
rosenbrock(x,p) =  (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p  = [1.0,100.0]

prob = OptimizationProblem(rosenbrock,x0,p)

using OptimizationOptimJL
sol = solve(prob,NelderMead())


using OptimizationBBO
prob = OptimizationProblem(rosenbrock, x0, p, lb = [-1.0,-1.0], ub = [1.0,1.0])
sol = solve(prob,BBO_adaptive_de_rand_1_bin_radiuslimited())
```

Notice that Optimization.jl is the core glue package that holds all of the common
pieces, but to solve the equations we need to use a solver package. Here, GalcticOptimJL
is for [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) and OptimizationBBO is for
[BlackBoxOptim.jl](https://github.com/robertfeldt/BlackBoxOptim.jl).

The output of the first optimization task (with the `NelderMead()` algorithm)
is given below:

```julia
sol = solve(prob,NelderMead())
u: 2-element Vector{Float64}:
 0.9999634355313174
 0.9999315506115275
```

The solution from the original solver can always be obtained via `original`:

```julia
julia> sol.original
 * Status: success

 * Candidate solution
    Final objective value:     3.525527e-09

 * Found with
    Algorithm:     Nelder-Mead

 * Convergence measures
    √(Σ(yᵢ-ȳ)²)/n ≤ 1.0e-08

 * Work counters
    Seconds run:   0  (vs limit Inf)
    Iterations:    60
    f(x) calls:    117
```

We can also explore other methods in a similar way:

```julia
using ForwardDiff
f = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
prob = OptimizationProblem(f, x0, p)
sol = solve(prob,BFGS())
```
For instance, the above optimization task produces the following output:

```julia
* Status: success

* Candidate solution
   Final objective value:     7.645684e-21

* Found with
   Algorithm:     BFGS

* Convergence measures
   |x - x'|               = 3.48e-07 ≰ 0.0e+00
   |x - x'|/|x'|          = 3.48e-07 ≰ 0.0e+00
   |f(x) - f(x')|         = 6.91e-14 ≰ 0.0e+00
   |f(x) - f(x')|/|f(x')| = 9.03e+06 ≰ 0.0e+00
   |g(x)|                 = 2.32e-09 ≤ 1.0e-08

* Work counters
   Seconds run:   0  (vs limit Inf)
   Iterations:    16
   f(x) calls:    53
   ∇f(x) calls:   53
```

```julia
 prob = OptimizationProblem(f, x0, p, lb = [-1.0,-1.0], ub = [1.0,1.0])
 sol = solve(prob, Fminbox(GradientDescent()))
```

The examples clearly demonstrate that Optimization.jl provides an intuitive
way of specifying optimization tasks and offers a relatively
easy access to a wide range of optimization algorithms.
