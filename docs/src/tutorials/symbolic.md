# Symbolic Problem Building with ModelingToolkit

!!! note
    
    This example uses the OptimizationOptimJL.jl package. See the [Optim.jl page](@ref optim)
    for details on the installation and usage.

[ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/) is a comprehensive system
for symbolic modeling in Julia. Allows for doing many manipulations before the solver phase,
such as detecting sparsity patterns, analytically solving parts of the model to reduce the
solving complexity, and more. One of the types of system types that it supports is
`OptimizationSystem`, i.e., the symbolic counterpart to `OptimizationProblem`. Let's demonstrate
how to use the `OptimizationSystem` to construct optimized `OptimizationProblem`s.

First we need to start by defining our symbolic variables, this is done as follows:

```@example modelingtoolkit
using ModelingToolkit, Optimization, OptimizationOptimJL

@variables x y
@parameters a b
```

We can now construct the `OptimizationSystem` by building a symbolic expression
for the loss function:

```@example modelingtoolkit
loss = (a - x)^2 + b * (y - x^2)^2
@named sys = OptimizationSystem(loss, [x, y], [a, b])
```

To turn it into a problem for numerical solutions, we need to specify what
our parameter values are and the initial conditions. This looks like:

```@example modelingtoolkit
u0 = [x => 1.0
      y => 2.0]
p = [a => 6.0
     b => 7.0]
```

And now we solve.

```@example modelingtoolkit
sys = complete(sys)
prob = OptimizationProblem(sys, u0, p, grad = true, hess = true)
solve(prob, Newton())
```

It provides many other features like auto-parallelism and sparsification too.
Plus, you can hierarchically nest systems to generate huge
optimization problems. Check out the
[ModelingToolkit.jl OptimizationSystem documentation](https://docs.sciml.ai/ModelingToolkit/stable/)
for more information.
