# Linear and Integer Optimization Problems

Below we show how to solve a linear optimization problem using the HiGHS optimizer.
This examples has been taken from the [JuMP documentation](https://jump.dev/JuMP.jl/stable/tutorials/linear/finance/#Short-term-financing) and a description of the problem can be found there.

```julia
using Optimization, OptimizationMOI, ModelingToolkit, HiGHS, LinearAlgebra

@variables u[1:5]
@variables v[1:3]
@variables w[1:5]
@variables m

cons = [
        u[1] + v[1] - w[1] ~ 150 # January
        u[2] + v[2] - w[2] - 1.01u[1] + 1.003w[1] ~ 100 # February
        u[3] + v[3] - w[3] - 1.01u[2] + 1.003w[2] ~ -200 # March
        u[4] - w[4] - 1.02v[1] - 1.01u[3] + 1.003w[3] ~ 200 # April
        u[5] - w[5] - 1.02v[2] - 1.01u[4] + 1.003w[4] ~ -50 # May
        -m - 1.02v[3] - 1.01u[5] + 1.003w[5] ~ -300 # June
    ]

@named optsys = OptimizationSystem(m, [u...,v...,w..., m], [], constraints = cons)
optprob = OptimizationProblem(optsys, [[u[i] => 0.0 for i in 1:5]... , [v[i] => 0.0 for i in 1:3]..., [w[i] => 0.0 for i in 1:5]..., m => 300.0]; lb = zeros(14), ub = vcat(ones(5).*100, fill(Inf, 9)), grad = true, hess = true, sense = Optimization.MaxSense)
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
