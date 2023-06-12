# Linear and Integer Optimization Problems

## Example: Short-Term Financing

Below we show how to solve a linear optimization problem using the HiGHS optimizer.
This example has been taken from the [JuMP documentation](https://jump.dev/JuMP.jl/stable/tutorials/linear/finance/#Short-term-financing).

Short-term cash commitments present an ongoing challenge for corporations[^1]. Let's explore an example scenario to understand this better:

Consider the following monthly net cash flow requirements, presented in thousands of dollars:

| Month         | Jan  | Feb  | Mar | Apr  | May | Jun |
|:------------- |:---- |:---- |:--- |:---- |:--- |:--- |
| Net Cash Flow | -150 | -100 | 200 | -200 | 50  | 300 |

To address these financial needs, our hypothetical company has access to various funding sources[^1]:

 1. A line of credit: The company can utilize a line of credit of up to $100,000, subject to a monthly interest rate of 1%.
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

@variables u[1:5] [bounds = (0, 100)]
@variables v[1:3] [bounds = (0, Inf)]
@variables w[1:5] [bounds = (0, Inf)]
@variables m [bounds = (0, Inf)]

cons = [u[1] + v[1] - w[1] ~ 150 # January
    u[2] + v[2] - w[2] - 1.01u[1] + 1.003w[1] ~ 100 # February
    u[3] + v[3] - w[3] - 1.01u[2] + 1.003w[2] ~ -200 # March
    u[4] - w[4] - 1.02v[1] - 1.01u[3] + 1.003w[3] ~ 200 # April
    u[5] - w[5] - 1.02v[2] - 1.01u[4] + 1.003w[4] ~ -50 # May
    -m - 1.02v[3] - 1.01u[5] + 1.003w[5] ~ -300]

@named optsys = OptimizationSystem(m, [u..., v..., w..., m], [], constraints = cons)
optprob = OptimizationProblem(optsys,
    vcat(fill(0.0, 13), 300.0);
    grad = true,
    hess = true,
    sense = Optimization.MaxSense)
sol = solve(optprob, HiGHS.Optimizer())
```

<!--
```julia
using Enzyme
objective(u, p) = u[14]

constraints(res,u,p) = res .= [
    u[1] + u[6] - u[9] # January
    ,u[2] + u[7] - u[10] - 1.01u[1] + 1.003u[9] # February
    ,u[3] + u[8] - u[11] - 1.01u[2] + 1.003u[10] # March
    ,u[4] - u[12] - 1.02u[6] - 1.01u[3] + 1.003u[11] # April
    ,u[5] - u[13] - 1.02u[7] - 1.01u[4] + 1.003u[12] # May
    ,-u[14] - 1.02u[8] - 1.01u[5] + 1.003u[13] # June
]

optf = OptimizationFunction(objective, Optimization.AutoModelingToolkit(), cons = constraints)
optprob = OptimizationProblem(optf, [zeros(13)..., 300]; lb = zeros(14), ub = vcat(ones(5).*100, fill(Inf, 9)), lcons = [150, 100, -200, 200, -50, -300], ucons = [150, 100, -200, 200, -50, -300], sense = Optimization.MaxSense)
sol = solve(optprob, HiGHS.Optimizer())
```
-->
## Mixed Integer Nonlinear Optimization

<!--
```julia
    using Juniper, Ipopt, HiGHS

    ModelingToolkit.@variables b[1:4] [bounds = (0, 1)]
    ModelingToolkit.@variables i[1:24]
    ModelingToolkit.@variables objvar

    for j in 1:16
        push!(vars, i[j])
        ModelingToolkit.setmetadata(i[j], ModelingToolkit.VariableBounds, (0, 5))
    end

    ModelingToolkit.setmetadata(i[21], ModelingToolkit.VariableBounds, (0, 15))
    ModelingToolkit.setmetadata(i[22], ModelingToolkit.VariableBounds, (0, 12))
    ModelingToolkit.setmetadata(i[23], ModelingToolkit.VariableBounds, (0, 9))
    ModelingToolkit.setmetadata(i[24], ModelingToolkit.VariableBounds, (0, 6))

    constraints = [
        -0.1 * b[1] - 0.2 * b[2] - 0.3 * b[3] - 0.4 * b[4] - i[21] - i[22] - i[23] -
        i[24] + objvar ~ 0.0
        ,9.0 ≲ i[21] * i[1] + i[22] * i[2] + i[23] * i[3] + i[24] * i[4]
        ,7.0 ≲ i[21] * i[5] + i[22] * i[6] + i[23] * i[7] + i[24] * i[8]
        ,12.0 ≲ i[21] * i[9] + i[22] * i[10] + i[23] * i[11] + i[24] * i[12]
        ,11.0 ≲ i[21] * i[13] + i[22] * i[14] + i[23] * i[15] + i[24] * i[16]
        ,-330 * i[1] - 360 * i[5] - 385 * i[9] - 415 * i[13] + 1700 * b[1] ≲ 0.0
        ,-330 * i[2] - 360 * i[6] - 385 * i[10] - 415 * i[14] + 1700 * b[2] ≲ 0.0
        ,-330 * i[3] - 360 * i[7] - 385 * i[11] - 415 * i[15] + 1700 * b[3] ≲ 0.0
        ,-330 * i[4] - 360 * i[8] - 385 * i[12] - 415 * i[16] + 1700 * b[4] ≲ 0.0
        ,330 * i[1] + 360 * i[5] + 385 * i[9] + 415 * i[13] - 1900 * b[1] ≲ 0.0
        ,330 * i[2] + 360 * i[6] + 385 * i[10] + 415 * i[14] - 1900 * b[2] ≲ 0.0
        ,330 * i[3] + 360 * i[7] + 385 * i[11] + 415 * i[15] - 1900 * b[3] ≲ 0.0
        ,330 * i[4] + 360 * i[8] + 385 * i[12] + 415 * i[16] - 1900 * b[4] ≲ 0.0
        ,-i[1] - i[5] - i[9] - i[13] + b[1] ≲ 0.0
        ,-i[2] - i[6] - i[10] - i[14] + b[2] ≲ 0.0
        ,-i[3] - i[7] - i[11] - i[15] + b[3] ≲ 0.0
        ,-i[4] - i[8] - i[12] - i[16] + b[4] ≲ 0.0
        ,i[1] + i[5] + i[9] + i[13] - 5 * b[1] ≲ 0.0
        ,i[2] + i[6] + i[10] + i[14] - 5 * b[2] ≲ 0.0
        ,i[3] + i[7] + i[11] + i[15] - 5 * b[3] ≲ 0.0
        ,i[4] + i[8] + i[12] + i[16] - 5 * b[4] ≲ 0.0
        ,b[1] - i[21] ≲ 0.0
        ,b[2] - i[22] ≲ 0.0
        ,b[3] - i[23] ≲ 0.0
        ,b[4] - i[24] ≲ 0.0
        ,-15 * b[1] + i[21] ≲ 0.0
        ,-12 * b[2] + i[22] ≲ 0.0
        ,-9 * b[3] + i[23] ≲ 0.0
        ,-6 * b[4] + i[24] ≲ 0.0
        ,8.0 ≲ i[21] + i[22] + i[23] + i[24]
        ,-b[1] + b[2] ≲ 0.0
        ,-b[2] + b[3] ≲ 0.0
        ,-b[3] + b[4] ≲ 0.0
        ,-i[21] + i[22] ≲ 0.0
        ,-i[22] + i[23] ≲ 0.0
        ,-i[23] + i[24] ≲ 0.0
    ]

    i_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 21, 22, 23, 24]
    vars = [b..., [i[j] for j in i_idxs]..., objvar]

    @named optsys = OptimizationSystem(objvar, vars, [], constraints = constraints)
    optprob = OptimizationProblem(optsys, vcat(ones(24), 5), int = vcat(fill(true, 24), false), grad = true,  cons_j = true, hess = true, cons_h = true)

    nl_solver = OptimizationMOI.MOI.OptimizerWithAttributes(Ipopt.Optimizer,
                                                            "print_level" => 0)
    mip_solver = OptimizationMOI.MOI.OptimizerWithAttributes(HiGHS.Optimizer,
                                                            "output_flag" => false
                                                            )
    minlp_solver = OptimizationMOI.MOI.OptimizerWithAttributes(Juniper.Optimizer,
                                                               "nl_solver" => nl_solver,
                                                               "mip_solver" => mip_solver)
     opt = OptimizationMOI.MOI.OptimizerWithAttributes(Alpine.Optimizer,
                                                                "minlp_solver" => minlp_solver,
                                                               "nl_solver" => nl_solver,
                                                               "mip_solver" => mip_solver)
    sol = solve(optprob, opt)
```
-->
<!--
```julia
using Juniper, Ipopt

    LB = [100, 1000, 1000, 10, 10, 10, 10, 10]
    UB = [10000, 10000, 10000, 1000, 1000, 1000, 1000, 1000]

    ModelingToolkit.@variables x[1:8]
    ModelingToolkit.@variables y[1:5] [bounds = (0, 1)]

    for j in 1:8
        ModelingToolkit.setmetadata(x[j], ModelingToolkit.VariableBounds, (LB[j], UB[j]))
    end

    function cons(res, u, p)
        x, y = u[1:8], u[9:13]
        res .= [0.0025 * (x[4] * y[1] + x[6] * y[2])
                        ,0.0025 * (x[5] - x[4] * y[1] + x[7])
                        , 0.01(x[8] - x[5] * y[3]),
                        100 * x[1] - x[1] * x[6] * y[1] + 833.33252 * x[4] * y[1]
                        ,x[2] * x[4] * y[4] - x[2] * x[7] - 1250 * x[4] + 1250 * x[5],
                        x[3] * x[5] * y[2] * y[5] - x[3] * x[8] * y[5] - 2500 * x[5] * y[1] * y[4] + 1250000,
                        y[1] * y[2] * y[3],
                        y[4] * y[5] - (y[2] * y[3]),
                        y[1] * y[5] - (y[2] * y[4]),
                        ]
    end
    lcons = fill(-Inf, 9)
    ucons = [1, 1, 1, 83333.333, 0, 0, 0, 0, 0]
    objective = (u,p) -> u[1] + u[2] + u[3]
    optf = OptimizationFunction(objective, Optimization.AutoForwardDiff(), cons = cons)
# @named optsys = OptimizationSystem(objective, [x..., y...], [], constraints = constraints)
optprob = OptimizationProblem(optf, vcat((LB + UB) ./ 2,zeros(5));lb = vcat(LB, fill(0, 5)), ub = vcat(UB, fill(1, 5)), lcons, ucons, int = vcat(fill(false, 8), fill(true, 5)))

nl_solver = OptimizationMOI.MOI.OptimizerWithAttributes(Ipopt.Optimizer,
                                                            "print_level" => 0)
mi_solver = OptimizationMOI.MOI.OptimizerWithAttributes(HiGHS.Optimizer,
                                                            "presolve" => "on",
                                                            "log_to_console" => false,
                                                            )
minlp_solver = OptimizationMOI.MOI.OptimizerWithAttributes(Juniper.Optimizer,
                                                            "nl_solver" => nl_solver,
                                                            "mip_solver" => mi_solver)
opt = OptimizationMOI.MOI.OptimizerWithAttributes(Alpine.Optimizer,
                                                            "minlp_solver" => minlp_solver,
                                                            "nlp_solver" => nl_solver,
                                                            "mip_solver" => mi_solver)
sol = solve(optprob, minlp_solver)
```
-->
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
