# [Using Equality and Inequality Constraints](@id constraints)

Multiple optimization packages available with the MathOptInterface and Optim's `IPNewton` solver can handle non-linear constraints.
Optimization.jl provides a simple interface to define the constraint as a julia function and then specify the bounds for the output
in `OptimizationFunction` to indicate if it's an equality or inequality constraint.

Let's define the Rosenbrock function as our objective function and consider the below inequalities as our constraints.

```math
\begin{aligned}

x_1^2 + x_2^2 \leq 0.8 \\

0.0 \leq x_1 * x_2 \leq 5.0
\end{aligned}
```

```@example constraints
using Optimization, OptimizationMOI, OptimizationOptimJL, Ipopt
using ForwardDiff, ModelingToolkit

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
_p = [1.0, 1.0]
```

Next we define the sum of squares and the product of the optimization variables as our constraint functions.

```@example constraints
cons(res, x, p) = (res .= [x[1]^2+x[2]^2, x[1]*x[2]])
```

We'll use the `IPNewton` solver from Optim to solve the problem.

```@example constraints
optprob = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff(), cons = cons)
prob = OptimizationProblem(optprob, x0, _p, lcons = [-Inf, -1.0], ucons = [0.8, 2.0])
sol = solve(prob, IPNewton())
```

Let's check that the constraints are satisfied and the objective is lower than at initial values to be sure.

```@example constraints
res = zeros(2)
cons(res, sol.u, _p)
res
```

```@example constraints
prob.f(sol.u, _p)
```

We can also use the Ipopt library with the OptimizationMOI package.

```@example constraints
sol = solve(prob, Ipopt.Optimizer())
```

```@example constraints
res = zeros(2)
cons(res, sol.u, _p)
res
```

```@example constraints
prob.f(sol.u, _p)
```

We can also use ModelingToolkit as our AD backend and generate symbolic derivatives and expression graph for the objective and constraints.

Let's modify the bounds to use the function as an equality constraint. The constraint now becomes -

```math
\begin{aligned}

x_1^2 + x_2^2 = 1.0 \\

x_1 * x_2 = 0.5
\end{aligned}
```

```@example constraints
optprob = OptimizationFunction(rosenbrock, Optimization.AutoModelingToolkit(), cons = cons)
prob = OptimizationProblem(optprob, x0, _p, lcons = [1.0, 0.5], ucons = [1.0, 0.5])
```

Below the AmplNLWriter.jl package is used with to use the Ipopt library to solve the problem.

```@example constraints
using AmplNLWriter, Ipopt_jll
sol = solve(prob, AmplNLWriter.Optimizer(Ipopt_jll.amplexe))
```

The constraints evaluate to 1.0 and 0.5 respectively as expected.

```@example constraints
res = zeros(2)
cons(res, sol.u, _p)
println(res)
```

## Constraints as Riemannian manifolds

Certain constraints can be efficiently handled using Riemannian optimization methods. This is the case when moving the solution on a manifold can be done quickly. See [here](https://juliamanifolds.github.io/Manifolds.jl/latest/index.html) for a (non-exhaustive) list of such manifolds, most prominent of them being spheres, hyperbolic spaces, Stiefel and Grassmann manifolds, symmetric positive definite matrices and various Lie groups.

Let's for example solve the Rosenbrock function optimization problem with just the spherical constraint. Note that the constraint isn't passed to `OptimizationFunction` but to the optimization method instead. Here we will use a [quasi-Newton optimizer based on the BFGS algorithm](https://manoptjl.org/stable/solvers/quasi_Newton/).

```@example manopt
using Optimization, Manopt, OptimizationManopt, Manifolds

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = [0.0, 1.0]
_p = [1.0, 100.0]

function rosenbrock_grad!(storage, x, p)
    # the first part can be computed using AD tools
    storage[1] = -2.0 * (p[1] - x[1]) - 4.0 * p[2] * (x[2] - x[1]^2) * x[1]
    storage[2] = 2.0 * p[2] * (x[2] - x[1]^2)
    # projection is needed because Riemannian optimizers expect
    # Riemannian gradients instead of Euclidean ones.
    project!(Sphere(1), storage, x, storage)
end

optprob = OptimizationFunction(rosenbrock; grad=rosenbrock_grad!)
opt = OptimizationManopt.QuasiNewtonOptimizer(Sphere(1))
prob = OptimizationProblem(optprob, x0, _p)
sol = Optimization.solve(prob, opt)
```

Note that currently `AutoForwardDiff` can't correctly compute the required Riemannian gradient for optimization. Riemannian optimizers require Riemannian gradients while `ForwardDiff.jl` returns normal Euclidean ones. Conversion from Euclidean to Riemannian gradients can be performed using the [`project`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/projections.html#Projections) function and (for certain manifolds) [`change_representer`](https://juliamanifolds.github.io/Manifolds.jl/stable/manifolds/metric.html#Manifolds.change_representer-Tuple{AbstractManifold,%20AbstractMetric,%20Any,%20Any}).

Note that the constraint is correctly preserved and the convergence is quite fast.

```@example manopt
println(norm(sol.u))
println(sol.original.stop.reason)
```

It is possible to mix Riemannian and equation-based constraints but it is currently a topic of active research. Manopt.jl offers solvers for such problems but they are not accessible via the Optimization.jl interface yet.
