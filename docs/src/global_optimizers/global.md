# Global Unconstrained Optimizers

These methods are performing global optimization on problems without
constraint equations.

## Recommended Methods

[Good benchmarks](https://github.com/jonathanBieler/BlackBoxOptimizationBenchmarking.jl)
Recommend `BBO()`.

## Optim.jl

- [`Optim.ParticleSwarm`](https://julianlsolvers.github.io/Optim.jl/stable/#algo/particle_swarm/): **Particle Swarm Optimization**

    * `solve(problem, ParticleSwarm(lower, upper, n_particles))`
    * `lower`/`upper` are vectors of lower/upper bounds respectively
    * `n_particles` is the number of particles in the swarm
    * defaults to: `lower = []`, `upper = []`, `n_particles = 0`

## BlackBoxOptim.jl

- [`BlackBoxOptim`](https://github.com/robertfeldt/BlackBoxOptim.jl): **(Meta-)heuristic/stochastic algorithms**
    * `solve(problem, BBO(method))`
    * the name of the method must be preceded by `:`, for example: `:de_rand_2_bin`
    * in GalacticOptim.jl, `BBO()` defaults to the recommended `adaptive_de_rand_1_bin_radiuslimited`
    * the available methods are listed [here](https://github.com/robertfeldt/BlackBoxOptim.jl#state-of-the-library)

## QuadDIRECT.jl

- [`QuadDIRECT`](https://github.com/timholy/QuadDIRECT.jl): **QuadDIRECT algorithm (inspired by DIRECT and MCS)**
    * `solve(problem, QuadDirect(), splits)`
    * `splits` is a list of 3-vectors with initial locations at which to evaluate the function (the values must be in strictly increasing order and lie within the specified bounds), for instance:
    ```julia
    prob = GalacticOptim.OptimizationProblem(f, x0, p, lb=[-3, -2], ub=[3, 2])
    solve(prob, QuadDirect(), splits = ([-2, 0, 2], [-1, 0, 1]))
    ```
    * also note that `QuadDIRECT` should (for now) be installed by doing: `] add https://github.com/timholy/QuadDIRECT.jl.git`

## Evolutionary.jl

- [`Evolutionary.GA`](https://wildart.github.io/Evolutionary.jl/stable/ga/): **Genetic Algorithm optimizer**

- [`Evolutionary.ES`](https://wildart.github.io/Evolutionary.jl/stable/es/): **Evolution Strategy algorithm**

- [`Evolutionary.CMAES`](https://wildart.github.io/Evolutionary.jl/stable/cmaes/): **Covariance Matrix Adaptation Evolution Strategy algorithm**

## CMAEvolutionStrategy.jl

- [`CMAEvolutionStrategy`](https://github.com/jbrea/CMAEvolutionStrategy.jl): **Covariance Matrix Adaptation Evolution Strategy algorithm**
