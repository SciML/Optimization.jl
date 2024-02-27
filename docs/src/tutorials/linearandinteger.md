# Linear and Integer Optimization Problems

## Example: Short-Term Financing

Below we show how to solve a linear optimization problem using the HiGHS optimizer.
This example has been taken from the [JuMP documentation](https://jump.dev/JuMP.jl/stable/tutorials/linear/finance/#Short-term-financing).

Short-term cash commitments present an ongoing challenge for corporations. Let's explore an example scenario to understand this better:

Consider the following monthly net cash flow requirements, presented in thousands of dollars:

| Month         | Jan  | Feb  | Mar | Apr  | May | Jun |
|:------------- |:---- |:---- |:--- |:---- |:--- |:--- |
| Net Cash Flow | -150 | -100 | 200 | -200 | 50  | 300 |

To address these financial needs, our hypothetical company has access to various funding sources:

 1. A line of credit: The company can utilize a line of credit of up to \$100,000, subject to a monthly interest rate of 1%.
 2. Commercial paper issuance: In any of the first three months, the company has the option to issue 90-day commercial paper with a cumulative interest rate of 2% for the three-month period.
 3. Surplus fund investment: Any excess funds can be invested, earning a monthly interest rate of 0.3%.

The objective is to determine the most cost-effective utilization of these funding sources, aiming to maximize the company's available funds by the end of June.

To model this problem, we introduce the following decision variables:

  - `u_i`: The amount drawn from the line of credit in month `i`.
  - `v_i`: The amount of commercial paper issued in month `i`.
  - `w_i`: The surplus funds in month `i`.

We need to consider the following constraints:

 1. Cash inflow must equal cash outflow for each month.
 2. Upper bounds must be imposed on `u_i` to ensure compliance with the line of credit limit.
 3. The decision variables `u_i`, `v_i`, and `w_i` must be non-negative.

The ultimate objective is to maximize the company's wealth in June, denoted by the variable `m`.

```@example linear
using Optimization, OptimizationMOI, ModelingToolkit, HiGHS, LinearAlgebra

@variables u[1:5] [bounds = (0.0, 100.0)]
@variables v[1:3] [bounds = (0.0, Inf)]
@variables w[1:5] [bounds = (0.0, Inf)]
@variables m [bounds = (0.0, Inf)]

cons = [u[1] + v[1] - w[1] ~ 150 # January
        u[2] + v[2] - w[2] - 1.01u[1] + 1.003w[1] ~ 100 # February
        u[3] + v[3] - w[3] - 1.01u[2] + 1.003w[2] ~ -200 # March
        u[4] - w[4] - 1.02v[1] - 1.01u[3] + 1.003w[3] ~ 200 # April
        u[5] - w[5] - 1.02v[2] - 1.01u[4] + 1.003w[4] ~ -50 # May
        -m - 1.02v[3] - 1.01u[5] + 1.003w[5] ~ -300]

@named optsys = OptimizationSystem(m, [u..., v..., w..., m], [], constraints = cons)
optsys = complete(optsys)
optprob = OptimizationProblem(optsys,
    vcat(fill(0.0, 13), 300.0);
    grad = true,
    hess = true,
    sense = Optimization.MaxSense)
sol = solve(optprob, HiGHS.Optimizer())
```

## Mixed Integer Nonlinear Optimization

We choose an example from the [Juniper.jl readme](https://github.com/lanl-ansi/Juniper.jl#use-with-jump) to demonstrate mixed integer nonlinear optimization with Optimization.jl. The problem can be stated as follows:

```math
\begin{aligned}

v &= [10,20,12,23,42] \\
w &= [12,45,12,22,21] \\

\text{maximize} \quad & \sum_{i=1}^5 v_i u_i \\

\text{subject to} \quad & \sum_{i=1}^5 w_i u_i^2 \leq 45 \\

& u_i \in \{0,1\} \quad \forall i \in \{1,2,3,4,5\}

\end{aligned}
```

which implies a maximization problem of binary variables $u_i$ with the objective as the dot product of `v` and `u` subject to a quadratic constraint on `u`.

```@example linear
using Juniper, Ipopt

v = [10, 20, 12, 23, 42]
w = [12, 45, 12, 22, 21]

objective = (u, p) -> (v = p[1:5]; dot(v, u))

cons = (res, u, p) -> (w = p[6:10]; res .= [sum(w[i] * u[i]^2 for i in 1:5)])

optf = OptimizationFunction(objective, Optimization.AutoModelingToolkit(), cons = cons)
optprob = OptimizationProblem(optf,
    zeros(5),
    vcat(v, w);
    sense = Optimization.MaxSense,
    lb = zeros(5),
    ub = ones(5),
    lcons = [-Inf],
    ucons = [45.0],
    int = fill(true, 5))

nl_solver = OptimizationMOI.MOI.OptimizerWithAttributes(Ipopt.Optimizer,
    "print_level" => 0)
minlp_solver = OptimizationMOI.MOI.OptimizerWithAttributes(Juniper.Optimizer,
    "nl_solver" => nl_solver)

sol = solve(optprob, minlp_solver)
```
