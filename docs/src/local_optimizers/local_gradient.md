# Local Gradient-Based Optimization

## Recommended Methods

`ADAM()` is a good default with decent convergence rate. `BFGS()` can
converge faster but is more prone to hitting bad local optima. `LBFGS()`
requires less memory than `BFGS` and thus can have better scaling.

## Flux.jl

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
    * Defaults:
        * `alphaguess = LineSearches.InitialHagerZhang()`
        * `linesearch = LineSearches.HagerZhang()`
        * `eta = 0.4`
        * `P = nothing`
        * `precondprep = (P, x) -> nothing`

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
    * Defaults:
        * `alphaguess = LineSearches.InitialPrevious()`
        * `linesearch = LineSearches.HagerZhang()`
        * `P = nothing`
        * `precondprep = (P, x) -> nothing`

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
    * Defaults:
        * `alphaguess = LineSearches.InitialStatic()`
        * `linesearch = LineSearches.HagerZhang()`
        * `initial_invH = nothing`
        * `initial_stepnorm = nothing`
        * `manifold = Flat()`

- [`Optim.LBFGS`](https://julianlsolvers.github.io/Optim.jl/stable/#algo/lbfgs/): **Limited-memory Broyden-Fletcher-Goldfarb-Shanno algorithm**
    * `m` is the number of history points
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
    * `manifold` specifies a (Riemannian) manifold on which the function is to be minimized (for more information, consult [this source](https://julianlsolvers.github.io/Optim.jl/stable/#algo/manifolds/))
        * available manifolds:
        * `Flat`
        * `Sphere`
        * `Stiefel`
        * meta-manifolds:
        * `PowerManifold`
        * `ProductManifold`
        * custom manifolds
    * `scaleinvH0`: whether to scale the initial Hessian approximation
    * Defaults:
        * `m = 10`
        * `alphaguess = LineSearches.InitialStatic()`
        * `linesearch = LineSearches.HagerZhang()`
        * `P = nothing`
        * `precondprep = (P, x) -> nothing`
        * `manifold = Flat()`
        * `scaleinvH0::Bool = true && (typeof(P) <: Nothing)`

- [`Optim.NGMRES`](https://julianlsolvers.github.io/Optim.jl/stable/#algo/ngmres/)
- [`Optim.OACCEL`](https://julianlsolvers.github.io/Optim.jl/stable/#algo/ngmres/)

### Optim Keyword Arguments

The following special keyword arguments can be used with Optim.jl optimizers:

* `x_tol`: Absolute tolerance in changes of the input vector `x`, in infinity norm. Defaults to `0.0`.
* `f_tol`: Relative tolerance in changes of the objective value. Defaults to `0.0`.
* `g_tol`: Absolute tolerance in the gradient, in infinity norm. Defaults to `1e-8`. For gradient free methods, this will control the main convergence tolerance, which is solver specific.
* `f_calls_limit`: A soft upper limit on the number of objective calls. Defaults to `0` (unlimited).
* `g_calls_limit`: A soft upper limit on the number of gradient calls. Defaults to `0` (unlimited).
* `h_calls_limit`: A soft upper limit on the number of Hessian calls. Defaults to `0` (unlimited).
* `allow_f_increases`: Allow steps that increase the objective value. Defaults to `false`. Note that, when setting this to `true`, the last iterate will be returned as the minimizer even if the objective increased.
* `iterations`: How many iterations will run before the algorithm gives up? Defaults to `1_000`.
* `store_trace`: Should a trace of the optimization algorithm's state be stored? Defaults to `false`.
* `show_trace`: Should a trace of the optimization algorithm's state be shown on `stdout`? Defaults to `false`.
* `extended_trace`: Save additional information. Solver dependent. Defaults to `false`.
* `trace_simplex`: Include the full simplex in the trace for `NelderMead`. Defaults to `false`.
* `show_every`: Trace output is printed every `show_every`th iteration.
* `time_limit`: A soft upper limit on the total run time. Defaults to `NaN` (unlimited).

## Ipopt.jl (MathOptInterface)

- [`Ipopt.Optimizer`](https://juliahub.com/docs/Ipopt/yMQMo/0.7.0/)
- Ipopt is a MathOptInterface optimizer, and thus its options are handled via
  `GalacticOptim.MOI.OptimizerWithAttributes(Ipopt.Optimizer, "option_name" => option_value, ...)`
- The full list of optimizer options can be found in the [Ipopt Documentation](https://coin-or.github.io/Ipopt/OPTIONS.html#OPTIONS_REF)

## NLopt.jl

NLopt.jl algorithms are chosen via `NLopt.Opt(:algname)`. Consult the
[NLopt Documentation](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/)
for more information on the algorithms. Possible algorithm names are:
