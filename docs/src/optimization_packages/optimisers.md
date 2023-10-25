# [Optimisers.jl](@id optimisers)

## Installation: OptimizationOptimisers.jl

To use this package, install the OptimizationOptimisers package:

```julia
import Pkg;
Pkg.add("OptimizationOptimisers");
```

In addition to the optimisation algorithms provided by the Optimisers.jl package this subpackage
also provides the Sophia optimisation algorithm.

## Local Unconstrained Optimizers

  - `Sophia`: Based on the recent paper https://arxiv.org/abs/2305.14342. It incorporates second order information
    in the form of the diagonal of the Hessian matrix hence avoiding the need to compute the complete hessian. It has been shown to converge faster than other first order methods such as Adam and SGD.
    
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
  - [`Optimisers.RAdam`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.OAdam): **Optimistic Adam optimizer**
    
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
  - [`Optimisers.ADAGrad`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.ADAGrad): **ADAGrad optimizer**
    
      + `solve(problem, ADAGrad(η))`
    
      + `η` is the learning rate
      + Defaults:
        
          * `η = 0.1`
  - [`Optimisers.ADADelta`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.ADADelta): **ADADelta optimizer**
    
      + `solve(problem, ADADelta(ρ))`
    
      + `ρ` is the gradient decay factor
      + Defaults:
        
          * `ρ = 0.9`
  - [`Optimisers.AMSGrad`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.ADAGrad): **AMSGrad optimizer**
    
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
  - [`Optimisers.ADABelief`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.ADABelief): **ADABelief variant of Adam**
    
      + `solve(problem, ADABelief(η, β::Tuple))`
    
      + `η` is the learning rate
      + `β::Tuple` is the decay of momentums
      + Defaults:
        
          * `η = 0.001`
          * `β::Tuple = (0.9, 0.999)`
