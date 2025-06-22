# SciPy.jl

[`SciPy`](https://scipy.org/) is a mature Python library that offers a rich family of optimization, root–finding and linear‐programming algorithms.  `OptimizationSciPy.jl` gives access to these routines through the unified `Optimization.jl` interface just like any native Julia optimizer.

!!! note
    `OptimizationSciPy.jl` relies on [`PythonCall`](https://github.com/cjdoris/PythonCall.jl).  A minimal Python distribution containing SciPy will be installed automatically on first use, so no manual Python set-up is required.

## Installation: OptimizationSciPy.jl

```julia
import Pkg
Pkg.add("OptimizationSciPy")
```

## Methods

Below is a catalogue of the solver families exposed by `OptimizationSciPy.jl` together with their convenience constructors.  All of them accept the usual keyword arguments `maxiters`, `maxtime`, `abstol`, `reltol`, `callback`, `progress` in addition to any SciPy-specific options (passed verbatim via keyword arguments to `solve`).

### Local Optimizer

#### Derivative-Free

  * `ScipyNelderMead()` – Simplex Nelder–Mead algorithm
  * `ScipyPowell()` – Powell search along conjugate directions
  * `ScipyCOBYLA()` – Linear approximation of constraints (supports nonlinear constraints)

#### Gradient-Based

  * `ScipyCG()` – Non-linear conjugate gradient
  * `ScipyBFGS()` – Quasi-Newton BFGS
  * `ScipyLBFGSB()` – Limited-memory BFGS with simple bounds
  * `ScipyNewtonCG()` – Newton-conjugate gradient (requires Hessian-vector products)
  * `ScipyTNC()` – Truncated Newton with bounds
  * `ScipySLSQP()` – Sequential least-squares programming (supports constraints)
  * `ScipyTrustConstr()` – Trust-region method for non-linear constraints

#### Hessian–Based / Trust-Region

  * `ScipyDogleg()`, `ScipyTrustNCG()`, `ScipyTrustKrylov()`, `ScipyTrustExact()` – Trust-region algorithms that optionally use or build Hessian information

### Global Optimizer

  * `ScipyDifferentialEvolution()` – Differential evolution (requires bounds)
  * `ScipyBasinhopping()` – Basin-hopping with local search
  * `ScipyDualAnnealing()` – Dual annealing simulated annealing
  * `ScipyShgo()` – Simplicial homology global optimisation (supports constraints)
  * `ScipyDirect()` – Deterministic `DIRECT` algorithm (requires bounds)
  * `ScipyBrute()` – Brute-force grid search (requires bounds)

### Linear & Mixed-Integer Programming

  * `ScipyLinprog("highs")` – LP solvers from the HiGHS project and legacy interior-point/simplex methods
  * `ScipyMilp()` – Mixed-integer linear programming via HiGHS branch-and-bound

### Root Finding & Non-Linear Least Squares *(experimental)*

Support for `ScipyRoot`, `ScipyRootScalar` and `ScipyLeastSquares` is available behind the scenes and will be documented once the APIs stabilise.

## Examples

### Unconstrained minimisation

```@example SciPy1
using Optimization, OptimizationSciPy

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p  = [1.0, 100.0]

f   = OptimizationFunction(rosenbrock, Optimization.AutoZygote())
prob = OptimizationProblem(f, x0, p)

sol = solve(prob, ScipyBFGS())
@show sol.objective   # ≈ 0 at optimum
```

### Constrained optimisation with COBYLA

```@example SciPy2
using Optimization, OptimizationSciPy

# Objective
obj(x, p) = (x[1] + x[2] - 1)^2

# Single non-linear constraint: x₁² + x₂² ≈ 1 (with small tolerance)
cons(res, x, p) = (res .= [x[1]^2 + x[2]^2 - 1.0])

x0   = [0.5, 0.5]
prob = OptimizationProblem(
    OptimizationFunction(obj; cons = cons),
    x0, nothing, lcons = [-1e-6], ucons = [1e-6])  # Small tolerance instead of exact equality

sol = solve(prob, ScipyCOBYLA())
@show sol.u, sol.objective
```

### Differential evolution (global) with custom options

```@example SciPy3
using Optimization, OptimizationSciPy, Random, Statistics
Random.seed!(123)

ackley(x, p) = -20exp(-0.2*sqrt(mean(x .^ 2))) - exp(mean(cos.(2π .* x))) + 20 + ℯ
x0 = zeros(2)                    # initial guess is ignored by DE
prob = OptimizationProblem(ackley, x0; lb = [-5.0, -5.0], ub = [5.0, 5.0])

sol = solve(prob, ScipyDifferentialEvolution(); popsize = 20, mutation = (0.5, 1))
@show sol.objective
```

## Passing solver-specific options

Any keyword that `Optimization.jl` does not interpret is forwarded directly to SciPy.  Refer to the [SciPy optimisation API](https://docs.scipy.org/doc/scipy/reference/optimize.html) for the exhaustive list of options.

```julia
sol = solve(prob, ScipyTrustConstr(); verbose = 3, maxiter = 10_000)
```

## Troubleshooting

The original Python result object is attached to the solution in the `original` field:

```julia
sol = solve(prob, ScipyBFGS())
println(sol.original)  # SciPy OptimizeResult
```

If SciPy raises an error it is re-thrown as a Julia `ErrorException` carrying the Python message, so look there first.

## Contributing

Bug reports and feature requests are welcome in the [Optimization.jl](https://github.com/SciML/Optimization.jl) issue tracker.  Pull requests that improve either the Julia wrapper or the documentation are highly appreciated. 

