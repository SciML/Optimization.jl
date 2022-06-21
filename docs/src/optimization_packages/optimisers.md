# [Optimisers.jl](@id optimisers)

## Installation: OptimizationFlux.jl

To use this package, install the OptimizationOptimisers package:

```julia
import Pkg; Pkg.add("OptimizationOptimisers")
```

## Local Unconstrained Optimizers 

- [`Optimisers.Descent`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.Descent): **Classic gradient descent optimizer with learning rate**

    * `solve(problem, Descent(η))`
    * `η` is the learning rate
    * Defaults:
        * `η = 0.1`

- [`Optimisers.Momentum`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.Momentum): **Classic gradient descent optimizer with learning rate and momentum**

    * `solve(problem, Momentum(η, ρ))`
    * `η` is the learning rate
    * `ρ` is the momentum
    * Defaults:
        * `η = 0.01`
        * `ρ = 0.9`

- [`Optimisers.Nesterov`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.Nesterov): **Gradient descent optimizer with learning rate and Nesterov momentum**

    * `solve(problem, Nesterov(η, ρ))`
    * `η` is the learning rate
    * `ρ` is the Nesterov momentum
    * Defaults:
        * `η = 0.01`
        * `ρ = 0.9`

- [`Optimisers.RMSProp`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.RMSProp): **RMSProp optimizer**

    * `solve(problem, RMSProp(η, ρ))`
    * `η` is the learning rate
    * `ρ` is the momentum
    * Defaults:
        * `η = 0.001`
        * `ρ = 0.9`

- [`Optimisers.Adam`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.Adam): **Adam optimizer**

    * `solve(problem, Adam(η, β::Tuple))`
    * `η` is the learning rate
    * `β::Tuple` is the decay of momentums
    * Defaults:
        * `η = 0.001`
        * `β::Tuple = (0.9, 0.999)`

- [`Optimisers.RAdam`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.RAdam): **Rectified Adam optimizer**

    * `solve(problem, RAdam(η, β::Tuple))`
    * `η` is the learning rate
    * `β::Tuple` is the decay of momentums
    * Defaults:
        * `η = 0.001`
        * `β::Tuple = (0.9, 0.999)`
- [`Optimisers.RAdam`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.OAdam): **Optimistic Adam optimizer**

    * `solve(problem, OAdam(η, β::Tuple))`
    * `η` is the learning rate
    * `β::Tuple` is the decay of momentums
    * Defaults:
        * `η = 0.001`
        * `β::Tuple = (0.5, 0.999)`

- [`Optimisers.AdaMax`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.AdaMax): **AdaMax optimizer**

    * `solve(problem, AdaMax(η, β::Tuple))`
    * `η` is the learning rate
    * `β::Tuple` is the decay of momentums
    * Defaults:
        * `η = 0.001`
        * `β::Tuple = (0.9, 0.999)`

- [`Optimisers.ADAGrad`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.ADAGrad): **ADAGrad optimizer**

    * `solve(problem, ADAGrad(η))`
    * `η` is the learning rate
    * Defaults:
        * `η = 0.1`

- [`Optimisers.ADADelta`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.ADADelta): **ADADelta optimizer**

    * `solve(problem, ADADelta(ρ))`
    * `ρ` is the gradient decay factor
    * Defaults:
        * `ρ = 0.9`

- [`Optimisers.AMSGrad`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.ADAGrad): **AMSGrad optimizer**

    * `solve(problem, AMSGrad(η, β::Tuple))`
    * `η` is the learning rate
    * `β::Tuple` is the decay of momentums
    * Defaults:
        * `η = 0.001`
        * `β::Tuple = (0.9, 0.999)`

- [`Optimisers.NAdam`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.NAdam): **Nesterov variant of the Adam optimizer**

    * `solve(problem, NAdam(η, β::Tuple))`
    * `η` is the learning rate
    * `β::Tuple` is the decay of momentums
    * Defaults:
        * `η = 0.001`
        * `β::Tuple = (0.9, 0.999)`

- [`Optimisers.AdamW`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.AdamW): **AdamW optimizer**

    * `solve(problem, AdamW(η, β::Tuple))`
    * `η` is the learning rate
    * `β::Tuple` is the decay of momentums
    * `decay` is the decay to weights
    * Defaults:
        * `η = 0.001`
        * `β::Tuple = (0.9, 0.999)`
        * `decay = 0`

- [`Optimisers.ADABelief`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.ADABelief): **ADABelief variant of Adam**

    * `solve(problem, ADABelief(η, β::Tuple))`
    * `η` is the learning rate
    * `β::Tuple` is the decay of momentums
    * Defaults:
        * `η = 0.001`
        * `β::Tuple = (0.9, 0.999)`
