# [Using Equality and Inequality Constraints](@id constraints)

Multiple optmization packages available with the MathOptInterface and Optim's `IPNewton` solver can handle non-linear constraints.
Optimization.jl provides a simple interface to define the constraint as a julia function and then specify the bounds for the output
in `OptimizationFunction` to indicate if it's an equality or inequality constraint.

Let's define the rosenbrock function as our objective function and consider the below inequalities as our constraints.

$$

x_1^2 + x_2^2 \leq 0.8

0.0 \leq x_1 * x_2 \leq 5.0
$$

```@example constraints
using Optimization, OptimizationMOI, OptimizationOptimJL, ForwardDiff, ModelingToolkit

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
prob.f(sol.u, _p)
```

We can also use the Ipopt library with the OptimizationMOI package.

```@example constraints
sol = solve(prob, Ipopt.Optimizer())
```

```@example constraints
res = zeros(2)
cons(res, sol.u, _p)
println(res)
prob.f(sol.u, _p)
```

We can also use ModelingToolkit as our AD backend and generate symbolic derivatives and expression graph for the objective and constraints.

Let's modify the bounds to use the function as an equality constraint. The constraint now becomes -


$$

x_1^2 + x_2^2 = 1.0

\leq x_1 * x_2 = 0.5
$$

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
