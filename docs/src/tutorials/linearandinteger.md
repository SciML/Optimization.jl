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
using OptimizationBase, OptimizationMOI, ModelingToolkit, HiGHS, LinearAlgebra, SciMLBase

# Define variables with bounds
# u: line of credit (up to 100k), v: commercial paper, w: surplus, m: final wealth
@variables begin
    u1 = 0.0, [bounds = (0.0, 100.0)]
    u2 = 0.0, [bounds = (0.0, 100.0)]
    u3 = 0.0, [bounds = (0.0, 100.0)]
    u4 = 0.0, [bounds = (0.0, 100.0)]
    u5 = 0.0, [bounds = (0.0, 100.0)]
    v1 = 0.0, [bounds = (0.0, Inf)]
    v2 = 0.0, [bounds = (0.0, Inf)]
    v3 = 0.0, [bounds = (0.0, Inf)]
    w1 = 0.0, [bounds = (0.0, Inf)]
    w2 = 0.0, [bounds = (0.0, Inf)]
    w3 = 0.0, [bounds = (0.0, Inf)]
    w4 = 0.0, [bounds = (0.0, Inf)]
    w5 = 0.0, [bounds = (0.0, Inf)]
    m = 300.0, [bounds = (0.0, Inf)]
end

cons = [u1 + v1 - w1 ~ 150 # January
        u2 + v2 - w2 - 1.01u1 + 1.003w1 ~ 100 # February
        u3 + v3 - w3 - 1.01u2 + 1.003w2 ~ -200 # March
        u4 - w4 - 1.02v1 - 1.01u3 + 1.003w3 ~ 200 # April
        u5 - w5 - 1.02v2 - 1.01u4 + 1.003w4 ~ -50 # May
        -m - 1.02v3 - 1.01u5 + 1.003w5 ~ -300]

@named optsys = OptimizationSystem(
    m, [u1, u2, u3, u4, u5, v1, v2, v3, w1, w2, w3, w4, w5, m], [], constraints = cons)
optsys = complete(optsys)

optprob = OptimizationProblem(optsys, [];
    grad = true,
    hess = true,
    sense = SciMLBase.MaxSense)
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
using Juniper, Ipopt, ADTypes, Symbolics

v = [10, 20, 12, 23, 42]
w = [12, 45, 12, 22, 21]

objective = (u, p) -> (v = p[1:5]; dot(v, u))

cons = (res, u, p) -> (w = p[6:10]; res .= [sum(w[i] * u[i]^2 for i in 1:5)])

optf = OptimizationFunction(objective, ADTypes.AutoSymbolics(), cons = cons)
optprob = OptimizationProblem(optf,
    zeros(5),
    vcat(v, w);
    sense = SciMLBase.MaxSense,
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
