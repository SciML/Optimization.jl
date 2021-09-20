# Flux.jl


- [`Flux.Optimise.Descent`](https://fluxml.ai/Flux.jl/stable/training/optimisers/#Flux.Optimise.Descent): **Classic gradient descent optimizer with learning rate**

    * `solve(problem, Descent(η))`
    * `η` is the learning rate
    * Defaults:
        * `η = 0.1`

- [`Flux.Optimise.Momentum`](https://fluxml.ai/Flux.jl/stable/training/optimisers/#Flux.Optimise.Momentum): **Classic gradient descent optimizer with learning rate and momentum**

    * `solve(problem, Momentum(η, ρ))`
    * `η` is the learning rate
    * `ρ` is the momentum
    * Defaults:
        * `η = 0.01`
        * `ρ = 0.9`

- [`Flux.Optimise.Nesterov`](https://fluxml.ai/Flux.jl/stable/training/optimisers/#Flux.Optimise.Nesterov): **Gradient descent optimizer with learning rate and Nesterov momentum**

    * `solve(problem, Nesterov(η, ρ))`
    * `η` is the learning rate
    * `ρ` is the Nesterov momentum
    * Defaults:
        * `η = 0.01`
        * `ρ = 0.9`

- [`Flux.Optimise.RMSProp`](https://fluxml.ai/Flux.jl/stable/training/optimisers/#Flux.Optimise.RMSProp): **RMSProp optimizer**

    * `solve(problem, RMSProp(η, ρ))`
    * `η` is the learning rate
    * `ρ` is the momentum
    * Defaults:
        * `η = 0.001`
        * `ρ = 0.9`

- [`Flux.Optimise.ADAM`](https://fluxml.ai/Flux.jl/stable/training/optimisers/#Flux.Optimise.ADAM): **ADAM optimizer**

    * `solve(problem, ADAM(η, β::Tuple))`
    * `η` is the learning rate
    * `β::Tuple` is the decay of momentums
    * Defaults:
        * `η = 0.001`
        * `β::Tuple = (0.9, 0.999)`

- [`Flux.Optimise.RADAM`](https://fluxml.ai/Flux.jl/stable/training/optimisers/#Flux.Optimise.RADAM): **Rectified ADAM optimizer**

    * `solve(problem, RADAM(η, β::Tuple))`
    * `η` is the learning rate
    * `β::Tuple` is the decay of momentums
    * Defaults:
        * `η = 0.001`
        * `β::Tuple = (0.9, 0.999)`

- [`Flux.Optimise.AdaMax`](https://fluxml.ai/Flux.jl/stable/training/optimisers/#Flux.Optimise.AdaMax): **AdaMax optimizer**

    * `solve(problem, AdaMax(η, β::Tuple))`
    * `η` is the learning rate
    * `β::Tuple` is the decay of momentums
    * Defaults:
        * `η = 0.001`
        * `β::Tuple = (0.9, 0.999)`

- [`Flux.Optimise.ADAGRad`](https://fluxml.ai/Flux.jl/stable/training/optimisers/#Flux.Optimise.ADAGrad): **ADAGrad optimizer**

    * `solve(problem, ADAGrad(η))`
    * `η` is the learning rate
    * Defaults:
        * `η = 0.1`

- [`Flux.Optimise.ADADelta`](https://fluxml.ai/Flux.jl/stable/training/optimisers/#Flux.Optimise.ADADelta): **ADADelta optimizer**

    * `solve(problem, ADADelta(ρ))`
    * `ρ` is the gradient decay factor
    * Defaults:
        * `ρ = 0.9`

- [`Flux.Optimise.AMSGrad`](https://fluxml.ai/Flux.jl/stable/training/optimisers/#Flux.Optimise.ADAGrad): **AMSGrad optimizer**

    * `solve(problem, AMSGrad(η, β::Tuple))`
    * `η` is the learning rate
    * `β::Tuple` is the decay of momentums
    * Defaults:
        * `η = 0.001`
        * `β::Tuple = (0.9, 0.999)`

- [`Flux.Optimise.NADAM`](https://fluxml.ai/Flux.jl/stable/training/optimisers/#Flux.Optimise.NADAM): **Nesterov variant of the ADAM optimizer**

    * `solve(problem, NADAM(η, β::Tuple))`
    * `η` is the learning rate
    * `β::Tuple` is the decay of momentums
    * Defaults:
        * `η = 0.001`
        * `β::Tuple = (0.9, 0.999)`

- [`Flux.Optimise.ADAMW`](https://fluxml.ai/Flux.jl/stable/training/optimisers/#Flux.Optimise.ADAMW): **ADAMW optimizer**

    * `solve(problem, ADAMW(η, β::Tuple))`
    * `η` is the learning rate
    * `β::Tuple` is the decay of momentums
    * `decay` is the decay to weights
    * Defaults:
        * `η = 0.001`
        * `β::Tuple = (0.9, 0.999)`
        * `decay = 0`
