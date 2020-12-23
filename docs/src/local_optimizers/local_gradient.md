# Local Gradient-Based Optimization

## Recommended Methods

`ADAM()` is a good default with decent convergence rate. `BFGS()` can
converge faster but is more prone to hitting bad local optima. `LBFGS()`
requires less memory than `BFGS` and thus can have better scaling.

## Flux.jl

- [`Flux.Optimise.Descent`](https://fluxml.ai/Flux.jl/stable/training/optimisers/#Flux.Optimise.Descent): **Classic gradient descent optimizer with learning rate**

    * `solve(problem, Descent(η))`
    * `η` is the learning rate
    * defaults to: `η = 0.1`

- [`Flux.Optimise.Momentum`](https://fluxml.ai/Flux.jl/stable/training/optimisers/#Flux.Optimise.Momentum): **Classic gradient descent optimizer with learning rate and momentum**

    * `solve(problem, Momentum(η, ρ))`
    * `η` is the learning rate
    * `ρ` is the momentum
    * defaults to: `η = 0.01, ρ = 0.9`

- [`Flux.Optimise.Nesterov`](https://fluxml.ai/Flux.jl/stable/training/optimisers/#Flux.Optimise.Nesterov): **Gradient descent optimizer with learning rate and Nesterov momentum**

    * `solve(problem, Nesterov(η, ρ))`
    * `η` is the learning rate
    * `ρ` is the Nesterov momentum
    * defaults to: `η = 0.01, ρ = 0.9`

- [`Flux.Optimise.RMSProp`](https://fluxml.ai/Flux.jl/stable/training/optimisers/#Flux.Optimise.RMSProp): **RMSProp optimizer**

    * `solve(problem, RMSProp(η, ρ))`
    * `η` is the learning rate
    * `ρ` is the momentum
    * defaults to: `η = 0.001, ρ = 0.9`

- [`Flux.Optimise.ADAM`](https://fluxml.ai/Flux.jl/stable/training/optimisers/#Flux.Optimise.ADAM): **ADAM optimizer**

    * `solve(problem, ADAM(η, β::Tuple))`
    * `η` is the learning rate
    * `β::Tuple` is the decay of momentums
    * defaults to: `η = 0.001, β::Tuple = (0.9, 0.999)`

- [`Flux.Optimise.RADAM`](https://fluxml.ai/Flux.jl/stable/training/optimisers/#Flux.Optimise.RADAM): **Rectified ADAM optimizer**

    * `solve(problem, RADAM(η, β::Tuple))`
    * `η` is the learning rate
    * `β::Tuple` is the decay of momentums
    * defaults to: `η = 0.001, β::Tuple = (0.9, 0.999)`

- [`Flux.Optimise.AdaMax`](https://fluxml.ai/Flux.jl/stable/training/optimisers/#Flux.Optimise.AdaMax): **AdaMax optimizer**

    * `solve(problem, AdaMax(η, β::Tuple))`
    * `η` is the learning rate
    * `β::Tuple` is the decay of momentums
    * defaults to: `η = 0.001, β::Tuple = (0.9, 0.999)`

- [`Flux.Optimise.ADAGRad`](https://fluxml.ai/Flux.jl/stable/training/optimisers/#Flux.Optimise.ADAGrad): **ADAGrad optimizer**

    * `solve(problem, ADAGrad(η))`
    * `η` is the learning rate
    * defaults to: `η = 0.1`

- [`Flux.Optimise.ADADelta`](https://fluxml.ai/Flux.jl/stable/training/optimisers/#Flux.Optimise.ADADelta): **ADADelta optimizer**

    * `solve(problem, ADADelta(ρ))`
    * `ρ` is the gradient decay factor
    * defaults to: `ρ = 0.9`

- [`Flux.Optimise.AMSGrad`](https://fluxml.ai/Flux.jl/stable/training/optimisers/#Flux.Optimise.ADAGrad): **AMSGrad optimizer**

    * `solve(problem, AMSGrad(η, β::Tuple))`
    * `η` is the learning rate
    * `β::Tuple` is the decay of momentums
    * defaults to: `η = 0.001, β::Tuple = (0.9, 0.999)`

- [`Flux.Optimise.NADAM`](https://fluxml.ai/Flux.jl/stable/training/optimisers/#Flux.Optimise.NADAM): **Nesterov variant of the ADAM optimizer**

    * `solve(problem, NADAM(η, β::Tuple))`
    * `η` is the learning rate
    * `β::Tuple` is the decay of momentums
    * defaults to: `η = 0.001, β::Tuple = (0.9, 0.999)`

- [`Flux.Optimise.ADAMW`](https://fluxml.ai/Flux.jl/stable/training/optimisers/#Flux.Optimise.ADAMW): **ADAMW optimizer**

    * `solve(problem, ADAMW(η, β::Tuple))`
    * `η` is the learning rate
    * `β::Tuple` is the decay of momentums
    * `decay` is the decay to weights
    * defaults to: `η = 0.001, β::Tuple = (0.9, 0.999), decay = 0`

## Optim.jl

- [`Optim.ConjugateGradient`](https://julianlsolvers.github.io/Optim.jl/stable/#algo/cg/): **Conjugate Gradient Descent**

    * `solve(problem, ConjugateGradient(alphaguess, linesearch, eta, P, precondprep))`
    * `alphaguess` computes the initial step length (for more information, consult [this source](https://github.com/JuliaNLSolvers/LineSearches.jl) and [this example](https://julianlsolvers.github.io/LineSearches.jl/latest/examples/generated/optim_initialstep.html))
        * available initial step length procedures:
        * `InitialPrevious`
        * `InitialStatic`
        * `InitialHagerZhang`
        * `InitialQuadratic`
        * `InitialConstantChange`
    * `linesearch` specifies the line search algorithm (for more information, consult [this source](https://github.com/JuliaNLSolvers/LineSearches.jl) and [this example](https://julianlsolvers.github.io/LineSearches.jl/latest/examples/generated/optim_linesearch.html))
        * available line search algorithms:
        * `HaegerZhang`
        * `MoreThuente`
        * `BackTracking`
        * `StrongWolfe`
        * `Static`
    * `eta` determines the next step direction
    * `P` is an optional preconditioner (for more information, see [this source](https://julianlsolvers.github.io/Optim.jl/v0.9.3/algo/precondition/))
    * `precondpred` is used to update `P` as the state variable `x` changes
    * defaults to:
    ```julia
    alphaguess = LineSearches.InitialHagerZhang(),
    linesearch = LineSearches.HagerZhang(),
    eta = 0.4,
    P = nothing,
    precondprep = (P, x) -> nothing
    ```

- [`Optim.GradientDescent`](https://julianlsolvers.github.io/Optim.jl/stable/#algo/gradientdescent/): **Gradient Descent (a quasi-Newton solver)**

    * `solve(problem, GradientDescent(alphaguess, linesearch, P, precondprep))`
    * `alphaguess` computes the initial step length (for more information, consult [this source](https://github.com/JuliaNLSolvers/LineSearches.jl) and [this example](https://julianlsolvers.github.io/LineSearches.jl/latest/examples/generated/optim_initialstep.html))
        * available initial step length procedures:
        * `InitialPrevious`
        * `InitialStatic`
        * `InitialHagerZhang`
        * `InitialQuadratic`
        * `InitialConstantChange`
    * `linesearch` specifies the line search algorithm (for more information, consult [this source](https://github.com/JuliaNLSolvers/LineSearches.jl) and [this example](https://julianlsolvers.github.io/LineSearches.jl/latest/examples/generated/optim_linesearch.html))
        * available line search algorithms:
        * `HaegerZhang`
        * `MoreThuente`
        * `BackTracking`
        * `StrongWolfe`
        * `Static`
    * `P` is an optional preconditioner (for more information, see [this source](https://julianlsolvers.github.io/Optim.jl/v0.9.3/algo/precondition/))
    * `precondpred` is used to update `P` as the state variable `x` changes
    * defaults to:
    ```julia
    alphaguess = LineSearches.InitialPrevious(),
    linesearch = LineSearches.HagerZhang(),
    P = nothing,
    precondprep = (P, x) -> nothing
    ```
- [`Optim.BFGS`](https://julianlsolvers.github.io/Optim.jl/stable/#algo/lbfgs/): **Broyden-Fletcher-Goldfarb-Shanno algorithm**

     * `solve(problem, BFGS(alpaguess, linesearch, initial_invH, initial_stepnorm, manifold))`
     * `alphaguess` computes the initial step length (for more information, consult [this source](https://github.com/JuliaNLSolvers/LineSearches.jl) and [this example](https://julianlsolvers.github.io/LineSearches.jl/latest/examples/generated/optim_initialstep.html))
         * available initial step length procedures:
         * `InitialPrevious`
         * `InitialStatic`
         * `InitialHagerZhang`
         * `InitialQuadratic`
         * `InitialConstantChange`
     * `linesearch` specifies the line search algorithm (for more information, consult [this source](https://github.com/JuliaNLSolvers/LineSearches.jl) and [this example](https://julianlsolvers.github.io/LineSearches.jl/latest/examples/generated/optim_linesearch.html))
         * available line search algorithms:
         * `HaegerZhang`
         * `MoreThuente`
         * `BackTracking`
         * `StrongWolfe`
         * `Static`
    * `initial_invH` specifies an optional initial matrix
    * `initial_stepnorm` determines that `initial_invH` is an identity matrix scaled by the value of `initial_stepnorm` multiplied by the sup-norm of the gradient at the initial point
    * `manifold` specifies a (Riemannian) manifold on which the function is to be minimized (for more information, consult [this source](https://julianlsolvers.github.io/Optim.jl/stable/#algo/manifolds/))
        * available manifolds:
        * `Flat`
        * `Sphere`
        * `Stiefel`
        * meta-manifolds:
        * `PowerManifold`
        * `ProductManifold`
        * custom manifolds
    * defaults to: `alphaguess = LineSearches.InitialStatic()`, `linesearch = LineSearches.HagerZhang()`, `initial_invH = nothing`, `initial_stepnorm = nothing`, `manifold = Flat()`

- [`Optim.LBFGS`](https://julianlsolvers.github.io/Optim.jl/stable/#algo/lbfgs/): **Limited-memory Broyden-Fletcher-Goldfarb-Shanno algorithm**

## NLopt.jl
