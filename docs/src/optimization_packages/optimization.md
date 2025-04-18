# Optimization.jl

There are some solvers that are available in the Optimization.jl package directly without the need to install any of the solver wrappers.

## Methods

  - `LBFGS`: The popular quasi-Newton method that leverages limited memory BFGS approximation of the inverse of the Hessian. Through a wrapper over the [L-BFGS-B](https://users.iems.northwestern.edu/%7Enocedal/lbfgsb.html) fortran routine accessed from the [LBFGSB.jl](https://github.com/Gnimuc/LBFGSB.jl/) package. It directly supports box-constraints.
    
    This can also handle arbitrary non-linear constraints through a Augmented Lagrangian method with bounds constraints described in 17.4 of Numerical Optimization by Nocedal and Wright. Thus serving as a general-purpose nonlinear optimization solver available directly in Optimization.jl.

  - `Sophia`: Based on the recent paper https://arxiv.org/abs/2305.14342. It incorporates second order information in the form of the diagonal of the Hessian matrix hence avoiding the need to compute the complete hessian. It has been shown to converge faster than other first order methods such as Adam and SGD.
    
      + `solve(problem, Sophia(; η, βs, ϵ, λ, k, ρ))`
    
      + `η` is the learning rate
      + `βs` are the decay of momentums
      + `ϵ` is the epsilon value
      + `λ` is the weight decay parameter
      + `k` is the number of iterations to re-compute the diagonal of the Hessian matrix
      + `ρ` is the momentum
      + Defaults:
        
          * `η = 0.001`
          * `βs = (0.9, 0.999)`
          * `ϵ = 1e-8`
          * `λ = 0.1`
          * `k = 10`
          * `ρ = 0.04`

## Examples

### Unconstrained rosenbrock problem

```@example L-BFGS

using Optimization, Zygote

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p = [1.0, 100.0]

optf = OptimizationFunction(rosenbrock, AutoZygote())
prob = Optimization.OptimizationProblem(optf, x0, p)
sol = solve(prob, Optimization.LBFGS())
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
res = solve(prob, Optimization.LBFGS(), maxiters = 100)
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
    state.iter % 25 == 1 && @show "Iteration: %5d, Loss: %.6e\n" state.iter l
    return l < 1e-1 ## Terminate if loss is small
end

function loss(ps, data)
    ypred = [smodel([data[1][i]], ps)[1] for i in eachindex(data[1])]
    return sum(abs2, ypred .- data[2])
end

optf = OptimizationFunction(loss, AutoZygote())
prob = OptimizationProblem(optf, ps_ca, data)

res = Optimization.solve(prob, Optimization.Sophia(), callback = callback)
```
