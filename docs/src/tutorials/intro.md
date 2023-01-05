# Basic usage

In this tutorial, we introduce the basics of Optimization.jl by showing
how to easily mix local optimizers from Optim.jl and global optimizers
from BlackBoxOptim.jl on the Rosenbrock equation. The simplest copy-pasteable
code to get started is the following:

```@example intro
# Import the package and define the problem to optimize
using Optimization
rosenbrock(u,p) =  (p[1] - u[1])^2 + p[2] * (u[2] - u[1]^2)^2
u0 = zeros(2)
p  = [1.0,100.0]

prob = OptimizationProblem(rosenbrock,u0,p)

# Import a solver package and solve the optimization problem
using OptimizationOptimJL
sol = solve(prob,NelderMead())

# Import a different solver package and solve the optimization problem a different way
using OptimizationBBO
prob = OptimizationProblem(rosenbrock, u0, p, lb = [-1.0,-1.0], ub = [1.0,1.0])
sol = solve(prob,BBO_adaptive_de_rand_1_bin_radiuslimited())
```

Notice that Optimization.jl is the core glue package that holds all the common
pieces, but to solve the equations, we need to use a solver package. Here, OptimizationOptimJL
is for [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) and OptimizationBBO is for
[BlackBoxOptim.jl](https://github.com/robertfeldt/BlackBoxOptim.jl).

The output of the first optimization task (with the `NelderMead()` algorithm)
is given below:

```@example intro
optf = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, u0, p, lb = [-1.0,-1.0], ub = [1.0,1.0])
sol = solve(prob,NelderMead())
```

The solution from the original solver can always be obtained via `original`:

```@example intro
sol.original
```

## Controlling Gradient Calculations (Automatic Differentiation)

Notice that both of the above methods were derivative-free methods, and thus no
gradients were required to do the optimization. However, often first order
optimization (i.e., using gradients) is much more efficient. Defining gradients
can be done in two ways. One way is to manually provide a gradient definition
in the `OptimizationFunction` constructor. However, the more convenient way
to obtain gradients is to provide an AD backend type. 

For example, let's now use the OptimizationOptimJL `BFGS` method to solve the same
problem. We will import the forward-mode automatic differentiation library
(`using ForwardDiff`) and then specify in the `OptimizationFunction` to
automatically construct the derivative functions using ForwardDiff.jl. This
looks like:

```@example intro
using ForwardDiff
optf = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, u0, p)
sol = solve(prob,BFGS())
```

We can inspect the `original` to see the statistics on the number of steps 
required and gradients computed:

```@example intro
sol.original
```

Sure enough, it's a lot less than the derivative-free methods!

However, the compute cost of forward-mode automatic differentiation scales
via the number of inputs, and thus as our optimization problem grows large it
slows down. To counteract this, for larger optimization problems (>100 state
variables) one normally would want to use reverse-mode automatic differentiation.
One common choice for reverse-mode automatic differentiation is Zygote.jl.
We can demonstrate this via:

```@example intro
using Zygote
optf = OptimizationFunction(rosenbrock, Optimization.AutoZygote())
prob = OptimizationProblem(optf, u0, p)
sol = solve(prob,BFGS())
```

## Setting Box Constraints

In many cases, one knows the potential bounds on the solution values. In
Optimization.jl, these can be supplied as the `lb` and `ub` arguments for
the lower bounds and upper bounds respectively, supplying a vector of
values with one per state variable. Let's now do our gradient-based
optimization with box constraints by rebuilding the OptimizationProblem:

```@example intro
prob = OptimizationProblem(optf, u0, p, lb = [-1.0,-1.0], ub = [1.0,1.0])
sol = solve(prob,BFGS())
```

For more information on handling constraints, in particular equality and
inequality constraints, take a look at the [constraints tutorial](@ref constraints).
