# Getting Started with Optimization.jl

In this tutorial, we introduce the basics of Optimization.jl by showing
how to easily mix local optimizers and global optimizers on the Rosenbrock equation.

The Rosenbrock equation is defined as follows:

```math
f(u,p) = (p_1 - u_1)^2 + p_2 * ( u_2 - u_1^2)^2
```

This is a parameterized optimization problem where we want to solve for the vector `u` s.t. `u` minimizes `f`.
The simplest copy-pasteable code using a quasi-Newton method (LBFGS) to solve the Rosenbrock problem is the following:

```@example intro
# Import the package and define the problem to optimize
using Optimization, OptimizationLBFGSB, Zygote
rosenbrock(u, p) = (p[1] - u[1])^2 + p[2] * (u[2] - u[1]^2)^2
u0 = zeros(2)
p = [1.0, 100.0]

optf = OptimizationFunction(rosenbrock, AutoZygote())
prob = OptimizationProblem(optf, u0, p)

sol = solve(prob, OptimizationLBFGSB.LBFGSB())
```

```@example intro
sol.u
```

```@example intro
sol.objective
```

Tada! That's how you do it. Now let's dive in a little more into what each part means and how to customize it all to your needs.

## Understanding the Solution Object

The solution object is a `SciMLBase.AbstractNoTimeSolution`, and thus it follows the
[SciMLBase Solution Interface for non-timeseries objects](https://docs.sciml.ai/SciMLBase/stable/interfaces/Solutions/) and is documented at the [solution type page](@ref solution).
However, for simplicity let's show a bit of it in action.

An optimization solution has an array interface so that it acts like the array that it solves for. This array syntax is shorthand for simply grabbing the solution `u`. For example:

```@example intro
sol[1] == sol.u[1]
```

```@example intro
Array(sol) == sol.u
```

`sol.objective` returns the final cost of the optimization. We can validate this by plugging it into our function:

```@example intro
rosenbrock(sol.u, p)
```

```@example intro
sol.objective
```

The `sol.retcode` gives us more information about the solution process.

```@example intro
sol.retcode
```

Here it says `ReturnCode.Success` which means that the solutuion successfully solved. We can learn more about the different return codes at
[the ReturnCode part of the SciMLBase documentation](https://docs.sciml.ai/SciMLBase/stable/interfaces/Solutions/#retcodes).

If we are interested about some of the statistics of the solving process, for example to help choose a better solver, we can investigate the `sol.stats`

```@example intro
sol.stats
```

That's just a bit of what's in there, check out the other pages for more information but now let's move onto customization.

## Import a different solver package and solve the problem

OptimizationOptimJL is a wrapper for [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) and OptimizationBBO is a wrapper for [BlackBoxOptim.jl](https://github.com/robertfeldt/BlackBoxOptim.jl).

First let's use the NelderMead a derivative free solver from Optim.jl:

```@example intro
using OptimizationOptimJL
sol = solve(prob, Optim.NelderMead())
```

BlackBoxOptim.jl offers derivative-free global optimization solvers that requrie the bounds to be set via `lb` and `ub` in the `OptimizationProblem`. Let's use the BBO_adaptive_de_rand_1_bin_radiuslimited() solver:

```@example intro
using OptimizationBBO
prob = OptimizationProblem(rosenbrock, u0, p, lb = [-1.0, -1.0], ub = [1.0, 1.0])
sol = solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited())
```

The solution from the original solver can always be obtained via `original`:

```@example intro
sol.original
```

## Defining the objective function

Optimization.jl assumes that your objective function takes two arguments `objective(x, p)`

 1. The optimization variables `x`.
 2. Other parameters `p`, such as hyper parameters of the cost function.
    If you have no “other parameters”, you can  safely disregard this argument. If your objective function is defined by someone else, you can create an anonymous function that just discards the extra parameters like this

```julia
obj = (x, p) -> objective(x) # Pass this function into OptimizationFunction
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
sol = solve(prob, OptimizationOptimJL.BFGS())
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
sol = solve(prob, OptimizationOptimJL.BFGS())
```

## Setting Box Constraints

In many cases, one knows the potential bounds on the solution values. In
Optimization.jl, these can be supplied as the `lb` and `ub` arguments for
the lower bounds and upper bounds respectively, supplying a vector of
values with one per state variable. Let's now do our gradient-based
optimization with box constraints by rebuilding the OptimizationProblem:

```@example intro
prob = OptimizationProblem(optf, u0, p, lb = [-1.0, -1.0], ub = [1.0, 1.0])
sol = solve(prob, OptimizationOptimJL.BFGS())
```

For more information on handling constraints, in particular equality and
inequality constraints, take a look at the [constraints tutorial](@ref constraints).
