# [OptimizationSophia.jl](@id sophia)

[`OptimizationSophia.jl`](https://github.com/SciML/Optimization.jl/tree/master/lib/OptimizationSophia) is a package that provides the Sophia optimizer for neural network training.

## Installation

To use this package, install the `OptimizationSophia` package:

```julia
using Pkg
Pkg.add("OptimizationSophia")
```

## Methods

```@docs
OptimizationSophia.Sophia
```

## Examples

### Train NN with Sophia

```@example Sophia
using OptimizationBase, OptimizationSophia, Lux, ADTypes, Zygote, MLUtils, Statistics, Random, ComponentArrays

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

optf = OptimizationFunction(loss, ADTypes.AutoZygote())
prob = OptimizationProblem(optf, ps_ca, data)

res = solve(prob, OptimizationSophia.Sophia(), callback = callback, epochs = 100)
```
