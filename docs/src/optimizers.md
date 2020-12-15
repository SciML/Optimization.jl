# Available optimizers by categories

The first entries give the original name and package of the optimizers (and are linked to their source documentation). The comments in the next line describe the usage of the optimizers in GalacticOptim.jl and list the default values of the constructors.

Note that the default value of the maximum number of iterations is `1000`. It can be changed by doing: `solve(problem, optimizer, maxiters = 2000)`.

## Global optimizers

- [`Optim.ParticleSwarm`](https://julianlsolvers.github.io/Optim.jl/stable/#algo/particle_swarm/): **Particle Swarm Optimization**

    * `solve(problem, ParticleSwarm(lower, upper, n_particles))`
    * `lower`/`upper` are vectors of lower/upper bounds respectively
    * `n_particles` is the number of particles in the swarm
    * defaults to: `lower = []`, `upper = []`, `n_particles = 0`

- [`BlackBoxOptim`](https://github.com/robertfeldt/BlackBoxOptim.jl): **(Meta-)heuristic/stochastic algorithms**
    * `solve(problem, BBO(method))`
    * the name of the method must be preceded by `:`, for example: `:de_rand_2_bin`
    * in GalacticOptim.jl, `BBO()` defaults to the recommended `adaptive_de_rand_1_bin_radiuslimited`
    * the available methods are listed [here](https://github.com/robertfeldt/BlackBoxOptim.jl#state-of-the-library)

- [`QuadDIRECT`](https://github.com/timholy/QuadDIRECT.jl): **QuadDIRECT algorithm (inspired by DIRECT and MCS)**
    * `solve(problem, QuadDirect(), splits)`
    * `splits` is a list of 3-vectors with initial locations at which to evaluate the function (the values must be in strictly increasing order and lie within the specified bounds), for instance:
    ```
    prob = GalacticOptim.OptimizationProblem(f, x0, p, lb=[-3, -2], ub=[3, 2])
    solve(prob, QuadDirect(), splits = ([-2, 0, 2], [-1, 0, 1]))
    ```
    * also note that `QuadDIRECT` should (for now) be installed by doing: `] add https://github.com/timholy/QuadDIRECT.jl.git`

- [`Evolutionary.GA`](https://wildart.github.io/Evolutionary.jl/stable/ga/): **Genetic Algorithm optimizer**

- [`Evolutionary.ES`](https://wildart.github.io/Evolutionary.jl/stable/es/): **Evolution Strategy algorithm**

- [`Evolutionary.CMAES`](https://wildart.github.io/Evolutionary.jl/stable/cmaes/): **Covariance Matrix Adaptation Evolution Strategy algorithm**

- [`CMAEvolutionStrategy`](https://github.com/jbrea/CMAEvolutionStrategy.jl): **Covariance Matrix Adaptation Evolution Strategy algorithm**

## Local gradient-based optimizers

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
    ```
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
    ```
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

## Local derivative-free optimizers

- [`Optim.NelderMead`](https://julianlsolvers.github.io/Optim.jl/stable/#algo/nelder_mead/): **Nelder-Mead optimizer**

    * `solve(problem, NelderMead(parameters, initial_simplex))`
    * `parameters = AdaptiveParameters()` or `parameters = FixedParameters()`
    * `initial_simplex = AffineSimplexer()`
    * defaults to: `parameters = AdaptiveParameters(), initial_simplex = AffineSimplexer()`

- [`Optim.SimulatedAnnealing`](https://julianlsolvers.github.io/Optim.jl/stable/#algo/simulated_annealing/): **Simulated Annealing**

    * `solve(problem, SimulatedAnnealing(neighbor, T, p))`
    * `neighbor` is a mutating function of the current and proposed `x`
    * `T` is a function of the current iteration that returns a temperature
    * `p` is a function of the current temperature
    * defaults to: `neighbor = default_neighbor!, T = default_temperature, p = kirkpatrick`

## Second-order optimizers

- [`Optim.Newton`](https://julianlsolvers.github.io/Optim.jl/stable/#algo/newton/)

## Constrained local optimization

- [`Optim.IPNewton`](https://julianlsolvers.github.io/Optim.jl/stable/#algo/ipnewton/)

## Constrained global optimization

- [`Optim.SAMIN`](https://julianlsolvers.github.io/Optim.jl/stable/#algo/samin/): **Simulated Annealing with bounds**

    * `solve(problem, SAMIN(nt, ns, rt, neps, f_tol, x_tol, coverage_ok, verbosity))`
    * defaults to:
    ```
    SAMIN(; nt::Int = 5  # reduce temperature every nt*ns*dim(x_init) evaluations
            ns::Int = 5  # adjust bounds every ns*dim(x_init) evaluations
            rt::T = 0.9  # geometric temperature reduction factor: when temp changes, new temp is t=rt*t
            neps::Int = 5  # number of previous best values the final result is compared to
            f_tol::T = 1e-12  # the required tolerance level for function value comparisons
            x_tol::T = 1e-6  # the required tolerance level for x
            coverage_ok::Bool = false,  # if false, increase temperature until initial parameter space is covered
            verbosity::Int = 0)  # scalar: 0, 1, 2 or 3 (default = 0).

    # copied verbatim from https://julianlsolvers.github.io/Optim.jl/stable/#algo/samin/#constructor
    ```
