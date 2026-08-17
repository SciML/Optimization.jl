# [Optimisers.jl](@id optimisers)

## Installation: OptimizationOptimisers.jl

To use this package, install the OptimizationOptimisers package:

```julia
import Pkg;
Pkg.add("OptimizationOptimisers");
```

In addition to the optimisation algorithms provided by the Optimisers.jl package this subpackage
also provides the Sophia optimisation algorithm.

## List of optimizers

  - [`Optimisers.Descent`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.Descent): **Classic gradient descent optimizer with learning rate**
    
      + `solve(problem, Optimisers.Descent(η))`
    
      + `η` is the learning rate
      + Defaults:
        
          * `η = 0.1`

  - [`Optimisers.Momentum`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.Momentum): **Classic gradient descent optimizer with learning rate and momentum**
    
      + `solve(problem, Optimisers.Momentum(η, ρ))`
    
      + `η` is the learning rate
      + `ρ` is the momentum
      + Defaults:
        
          * `η = 0.01`
          * `ρ = 0.9`
  - [`Optimisers.Nesterov`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.Nesterov): **Gradient descent optimizer with learning rate and Nesterov momentum**
    
      + `solve(problem, Optimisers.Nesterov(η, ρ))`
    
      + `η` is the learning rate
      + `ρ` is the Nesterov momentum
      + Defaults:
        
          * `η = 0.01`
          * `ρ = 0.9`
  - [`Optimisers.RMSProp`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.RMSProp): **RMSProp optimizer**
    
      + `solve(problem, Optimisers.RMSProp(η, ρ))`
    
      + `η` is the learning rate
      + `ρ` is the momentum
      + Defaults:
        
          * `η = 0.001`
          * `ρ = 0.9`
  - [`Optimisers.Adam`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.Adam): **Adam optimizer**
    
      + `solve(problem, Optimisers.Adam(η, β::Tuple))`
    
      + `η` is the learning rate
      + `β::Tuple` is the decay of momentums
      + Defaults:
        
          * `η = 0.001`
          * `β::Tuple = (0.9, 0.999)`
  - [`Optimisers.RAdam`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.RAdam): **Rectified Adam optimizer**
    
      + `solve(problem, Optimisers.RAdam(η, β::Tuple))`
    
      + `η` is the learning rate
      + `β::Tuple` is the decay of momentums
      + Defaults:
        
          * `η = 0.001`
          * `β::Tuple = (0.9, 0.999)`
  - [`Optimisers.OAdam`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.OAdam): **Optimistic Adam optimizer**
    
      + `solve(problem, Optimisers.OAdam(η, β::Tuple))`
    
      + `η` is the learning rate
      + `β::Tuple` is the decay of momentums
      + Defaults:
        
          * `η = 0.001`
          * `β::Tuple = (0.5, 0.999)`
  - [`Optimisers.AdaMax`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.AdaMax): **AdaMax optimizer**
    
      + `solve(problem, Optimisers.AdaMax(η, β::Tuple))`
    
      + `η` is the learning rate
      + `β::Tuple` is the decay of momentums
      + Defaults:
        
          * `η = 0.001`
          * `β::Tuple = (0.9, 0.999)`
  - [`Optimisers.ADAGrad`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.ADAGrad): **ADAGrad optimizer**
    
      + `solve(problem, Optimisers.ADAGrad(η))`
    
      + `η` is the learning rate
      + Defaults:
        
          * `η = 0.1`
  - [`Optimisers.ADADelta`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.ADADelta): **ADADelta optimizer**
    
      + `solve(problem, Optimisers.ADADelta(ρ))`
    
      + `ρ` is the gradient decay factor
      + Defaults:
        
          * `ρ = 0.9`
  - [`Optimisers.AMSGrad`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.ADAGrad): **AMSGrad optimizer**
    
      + `solve(problem, Optimisers.AMSGrad(η, β::Tuple))`
    
      + `η` is the learning rate
      + `β::Tuple` is the decay of momentums
      + Defaults:
        
          * `η = 0.001`
          * `β::Tuple = (0.9, 0.999)`
  - [`Optimisers.NAdam`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.NAdam): **Nesterov variant of the Adam optimizer**
    
      + `solve(problem, Optimisers.NAdam(η, β::Tuple))`
    
      + `η` is the learning rate
      + `β::Tuple` is the decay of momentums
      + Defaults:
        
          * `η = 0.001`
          * `β::Tuple = (0.9, 0.999)`
  - [`Optimisers.AdamW`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.AdamW): **AdamW optimizer**
    
      + `solve(problem, Optimisers.AdamW(η, β::Tuple))`
    
      + `η` is the learning rate
      + `β::Tuple` is the decay of momentums
      + `decay` is the decay to weights
      + Defaults:
        
          * `η = 0.001`
          * `β::Tuple = (0.9, 0.999)`
          * `decay = 0`
  - [`Optimisers.ADABelief`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.ADABelief): **ADABelief variant of Adam**
    
      + `solve(problem, Optimisers.ADABelief(η, β::Tuple))`
    
      + `η` is the learning rate
      + `β::Tuple` is the decay of momentums
      + Defaults:
        
          * `η = 0.001`
          * `β::Tuple = (0.9, 0.999)`
