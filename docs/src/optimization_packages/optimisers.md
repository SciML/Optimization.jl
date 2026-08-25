# [Optimisers.jl](@id optimisers)

## Installation: OptimizationOptimisers.jl

To use this package, install the OptimizationOptimisers package:

```julia
import Pkg;
Pkg.add("OptimizationOptimisers");
```

In addition to the optimisation algorithms provided by the Optimisers.jl package this subpackage
also provides the Sophia optimisation algorithm.

## Reexported Optimisers.jl API

`using OptimizationOptimisers` brings Optimisers.jl's rules into scope, so that
`solve(prob, Adam(0.05))` works without a separate `using Optimisers`. These names are
owned and documented by [Optimisers.jl](https://fluxml.ai/Optimisers.jl/dev/api/); this
package only re-exports them.

  - Gradient descent rules: `Descent`, `Momentum`, `Nesterov`, `Rprop`
  - Adaptive rules: `RMSProp`, `Adam`, `RAdam`, `AdaMax`, `OAdam`, `AdaGrad`,
    `AdaDelta`, `AMSGrad`, `NAdam`, `AdamW`, `AdaBelief`, `Lion`
  - Gradient modifiers and combinators: `ClipGrad`, `ClipNorm`, `WeightDecay`,
    `SignDecay`, `AccumGrad`, `OptimiserChain`
  - The `Optimisers` module itself; the rule supertype is `Optimisers.AbstractRule`

Optimisers' own optimiser-driving interface — `Optimisers.setup`, `Optimisers.update`,
`Optimisers.update!`, `Optimisers.apply!`, `Optimisers.destructure`,
`Optimisers.trainables` — is deliberately *not* re-exported: `solve` drives the rule for
you, and `Optimisers.init` would collide with the SciML `init`.

Anything else from Optimisers.jl must be imported from Optimisers directly.

## List of optimizers

  - [`Optimisers.Descent`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.Descent): **Classic gradient descent optimizer with learning rate**
    
      + `solve(problem, Descent(η))`
    
      + `η` is the learning rate
      + Defaults:
        
          * `η = 0.1`

  - [`Optimisers.Momentum`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.Momentum): **Classic gradient descent optimizer with learning rate and momentum**
    
      + `solve(problem, Momentum(η, ρ))`
    
      + `η` is the learning rate
      + `ρ` is the momentum
      + Defaults:
        
          * `η = 0.01`
          * `ρ = 0.9`
  - [`Optimisers.Nesterov`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.Nesterov): **Gradient descent optimizer with learning rate and Nesterov momentum**
    
      + `solve(problem, Nesterov(η, ρ))`
    
      + `η` is the learning rate
      + `ρ` is the Nesterov momentum
      + Defaults:
        
          * `η = 0.01`
          * `ρ = 0.9`
  - [`Optimisers.RMSProp`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.RMSProp): **RMSProp optimizer**
    
      + `solve(problem, RMSProp(η, ρ))`
    
      + `η` is the learning rate
      + `ρ` is the momentum
      + Defaults:
        
          * `η = 0.001`
          * `ρ = 0.9`
  - [`Optimisers.Adam`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.Adam): **Adam optimizer**
    
      + `solve(problem, Adam(η, β::Tuple))`
    
      + `η` is the learning rate
      + `β::Tuple` is the decay of momentums
      + Defaults:
        
          * `η = 0.001`
          * `β::Tuple = (0.9, 0.999)`
  - [`Optimisers.RAdam`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.RAdam): **Rectified Adam optimizer**
    
      + `solve(problem, RAdam(η, β::Tuple))`
    
      + `η` is the learning rate
      + `β::Tuple` is the decay of momentums
      + Defaults:
        
          * `η = 0.001`
          * `β::Tuple = (0.9, 0.999)`
  - [`Optimisers.OAdam`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.OAdam): **Optimistic Adam optimizer**
    
      + `solve(problem, OAdam(η, β::Tuple))`
    
      + `η` is the learning rate
      + `β::Tuple` is the decay of momentums
      + Defaults:
        
          * `η = 0.001`
          * `β::Tuple = (0.5, 0.999)`
  - [`Optimisers.AdaMax`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.AdaMax): **AdaMax optimizer**
    
      + `solve(problem, AdaMax(η, β::Tuple))`
    
      + `η` is the learning rate
      + `β::Tuple` is the decay of momentums
      + Defaults:
        
          * `η = 0.001`
          * `β::Tuple = (0.9, 0.999)`
  - [`Optimisers.AdaGrad`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.AdaGrad): **AdaGrad optimizer**
    
      + `solve(problem, AdaGrad(η))`
    
      + `η` is the learning rate
      + Defaults:
        
          * `η = 0.1`
  - [`Optimisers.AdaDelta`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.AdaDelta): **AdaDelta optimizer**
    
      + `solve(problem, AdaDelta(ρ))`
    
      + `ρ` is the gradient decay factor
      + Defaults:
        
          * `ρ = 0.9`
  - [`Optimisers.AMSGrad`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.AMSGrad): **AMSGrad optimizer**
    
      + `solve(problem, AMSGrad(η, β::Tuple))`
    
      + `η` is the learning rate
      + `β::Tuple` is the decay of momentums
      + Defaults:
        
          * `η = 0.001`
          * `β::Tuple = (0.9, 0.999)`
  - [`Optimisers.NAdam`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.NAdam): **Nesterov variant of the Adam optimizer**
    
      + `solve(problem, NAdam(η, β::Tuple))`
    
      + `η` is the learning rate
      + `β::Tuple` is the decay of momentums
      + Defaults:
        
          * `η = 0.001`
          * `β::Tuple = (0.9, 0.999)`
  - [`Optimisers.AdamW`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.AdamW): **AdamW optimizer**
    
      + `solve(problem, AdamW(η, β::Tuple))`
    
      + `η` is the learning rate
      + `β::Tuple` is the decay of momentums
      + `decay` is the decay to weights
      + Defaults:
        
          * `η = 0.001`
          * `β::Tuple = (0.9, 0.999)`
          * `decay = 0`
  - [`Optimisers.AdaBelief`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.AdaBelief): **AdaBelief variant of Adam**
    
      + `solve(problem, AdaBelief(η, β::Tuple))`
    
      + `η` is the learning rate
      + `β::Tuple` is the decay of momentums
      + Defaults:
        
          * `η = 0.001`
          * `β::Tuple = (0.9, 0.999)`
