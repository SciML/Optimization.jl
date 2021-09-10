# Global Unconstrained Optimizers

These methods are performing global optimization on problems without
constraint equations. Note that each of these optimizers do support bounds
constraints set by `lb` and `ub` in the `OptimizationProblem` construction.

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

## NLopt.jl

NLopt.jl algorithms are chosen via `NLopt.Opt(:algname, nparameter)` or `NLO(:algname)` where `nparameter` is the number of parameters to be optimized . Consult the
[NLopt Documentation](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/)
for more information on the algorithms. Possible algorithm names are:

* `:GN_DIRECT`
* `:GN_DIRECT_L`
* `:GN_CRS2_LM`
* `:G_MLSL_LDS`
* `:GD_STOGO`
* `:GN_ESCH`

The following optimizer parameters can be set as `kwargs`:

* `stopval`
* `ftol_rel`
* `ftol_abs`
* `xtol_rel`
* `xtol_abs`
* `constrtol_abs`
* `maxeval`
* `maxtime`
* `initial_step`
* `population`
* `vector_storage`

Running an optimisation with `:GN_DIRECT` with setting the number iterations via the common argument `maxiters` and `NLopt.jl`-specific parameters such as the maximum time to perform the optimisation via `maxtime`:
```julia
rosenbrock(x, p) =  (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p  = [1.0, 100.0]
f = OptimizationFunction(rosenbrock, GalacticOptim.AutoForwardDiff())
prob = OptimizationProblem(f, x0, p, lb = [-1.0,-1.0], ub = [1.0,1.0])
sol = solve(prob, NLO(:GN_DIRECT), maxiters=100000, maxtime=1000.0)
```

For algorithms such as `:G_MLSL` `:G_MLSL_LDS` also a local optimiser needs to be chosen which is done via `NLopt.Opt(:algname, nparameter)` or `NLO(:algname)` passed to the `local_method` argument of `solve`. The number iterations for the local optimiser are set via `local_maxiters` and the local optimiser parameters as listed above are set via a `NamedTuple` passed to the `local_options` argument of solve.

Running an optimisation with `:G_MLSL_LDS` with setting the number iterations via the common argument `maxiters` and `NLopt.jl`-specific parameters such the number of local optimisation and maximum time to perform the optimisation via `population` and `maxtime` respectively and additionally setting the local optimizer to `:LN_NELDERMEAD`. The local optimizer maximum iterations are set via `local_maxiters`:

```julia
rosenbrock(x, p) =  (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p  = [1.0, 100.0]
f = OptimizationFunction(rosenbrock, GalacticOptim.AutoForwardDiff())
prob = OptimizationProblem(f, x0, p, lb = [-1.0,-1.0], ub = [1.0,1.0])
sol = solve(prob, NLO(:G_MLSL_LDS), local_method = NLO(:LN_NELDERMEAD), local_maxiters=10000, maxiters=10000, maxtime=1000.0, population=10)
```
