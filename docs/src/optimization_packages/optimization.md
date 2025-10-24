# Optimization.jl

There are some solvers that are available in the Optimization.jl package directly without the need to install any of the solver wrappers.

## Methods

  - `LBFGS`: The popular quasi-Newton method that leverages limited memory BFGS approximation of the inverse of the Hessian. Through a wrapper over the [L-BFGS-B](https://users.iems.northwestern.edu/%7Enocedal/lbfgsb.html) fortran routine accessed from the [LBFGSB.jl](https://github.com/Gnimuc/LBFGSB.jl/) package. It directly supports box-constraints.

    This can also handle arbitrary non-linear constraints through a Augmented Lagrangian method with bounds constraints described in 17.4 of Numerical Optimization by Nocedal and Wright. Thus serving as a general-purpose nonlinear optimization solver available directly in Optimization.jl.

```@docs
Optimization.Sophia
```

## Examples

### Unconstrained rosenbrock problem

```@example L-BFGS

using Optimization, OptimizationLBFGSB, Zygote

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p = [1.0, 100.0]

optf = OptimizationFunction(rosenbrock, AutoZygote())
prob = Optimization.OptimizationProblem(optf, x0, p)
sol = solve(prob, LBFGS())
```

### With nonlinear and bounds constraints

```@example L-BFGS

function con2_c(res, x, p)
    res .= [x[1]^2 + x[2]^2, (x[2] * sin(x[1]) + x[1]) - 5]
end

optf = OptimizationFunction(rosenbrock, AutoZygote(), cons = con2_c)
prob = OptimizationProblem(optf, x0, p, lcons = [1.0, -Inf],
    ucons = [1.0, 0.0], lb = [-1.0, -1.0],
    ub = [1.0, 1.0])
res = solve(prob, LBFGS(), maxiters = 100)
```

### Train NN with Sophia

```@example Sophia

using Optimization, Lux, Zygote, MLUtils, Statistics, Plots, Random, ComponentArrays

x = rand(10000)
y = sin.(x)
data = MLUtils.DataLoader((x, y), batchsize = 100)

# Define the neural network
model = Chain(Dense(1, 32, tanh), Dense(32, 1))
ps, st = Lux.setup(Random.default_rng(), model)
ps_ca = ComponentArray(ps)
smodel = StatefulLuxLayer{true}(model, nothing, st)

function callback(state, l)
    state.iter % 25 == 1 && @show "Iteration: $(state.iter), Loss: $l"
    return l < 1e-1 ## Terminate if loss is small
end

function loss(ps, data)
    x_batch, y_batch = data
    ypred = [smodel([x_batch[i]], ps)[1] for i in eachindex(x_batch)]
    return sum(abs2, ypred .- y_batch)
end

optf = OptimizationFunction(loss, AutoZygote())
prob = OptimizationProblem(optf, ps_ca, data)

res = Optimization.solve(prob, Optimization.Sophia(), callback = callback, epochs = 100)
```
