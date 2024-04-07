# NLopt.jl

[`NLopt`](https://github.com/JuliaOpt/NLopt.jl) is Julia package interfacing to the free/open-source [`NLopt library`](http://ab-initio.mit.edu/nlopt) which implements many optimization methods both global and local [`NLopt Documentation`](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/).

## Installation: OptimizationNLopt.jl

To use this package, install the OptimizationNLopt package:

```julia
import Pkg;
Pkg.add("OptimizationNLopt");
```

## Methods

`NLopt.jl` algorithms are chosen either via `NLopt.Opt(:algname, nstates)` where nstates is the number of states to be optimized,
but preferably via `NLopt.AlgorithmName()` where `AlgorithmName can be one of the following:

  - `NLopt.GN_DIRECT()`
  - `NLopt.GN_DIRECT_L()`
  - `NLopt.GN_DIRECT_L_RAND()`
  - `NLopt.GN_DIRECT_NOSCAL()`
  - `NLopt.GN_DIRECT_L_NOSCAL()`
  - `NLopt.GN_DIRECT_L_RAND_NOSCAL()`
  - `NLopt.GN_ORIG_DIRECT()`
  - `NLopt.GN_ORIG_DIRECT_L()`
  - `NLopt.GD_STOGO()`
  - `NLopt.GD_STOGO_RAND()`
  - `NLopt.LD_LBFGS()`
  - `NLopt.LN_PRAXIS()`
  - `NLopt.LD_VAR1()`
  - `NLopt.LD_VAR2()`
  - `NLopt.LD_TNEWTON()`
  - `NLopt.LD_TNEWTON_RESTART()`
  - `NLopt.LD_TNEWTON_PRECOND()`
  - `NLopt.LD_TNEWTON_PRECOND_RESTART()`
  - `NLopt.GN_CRS2_LM()`
  - `NLopt.GN_MLSL()`
  - `NLopt.GD_MLSL()`
  - `NLopt.GN_MLSL_LDS()`
  - `NLopt.GD_MLSL_LDS()`
  - `NLopt.LD_MMA()`
  - `NLopt.LN_COBYLA()`
  - `NLopt.LN_NEWUOA()`
  - `NLopt.LN_NEWUOA_BOUND()`
  - `NLopt.LN_NELDERMEAD()`
  - `NLopt.LN_SBPLX()`
  - `NLopt.LN_AUGLAG()`
  - `NLopt.LD_AUGLAG()`
  - `NLopt.LN_AUGLAG_EQ()`
  - `NLopt.LD_AUGLAG_EQ()`
  - `NLopt.LN_BOBYQA()`
  - `NLopt.GN_ISRES()`
  - `NLopt.AUGLAG()`
  - `NLopt.AUGLAG_EQ()`
  - `NLopt.G_MLSL()`
  - `NLopt.G_MLSL_LDS()`
  - `NLopt.LD_SLSQP()`
  - `NLopt.LD_CCSAQ()`
  - `NLopt.GN_ESCH()`
  - `NLopt.GN_AGS()`

See the [`NLopt Documentation`](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/) for more details on each optimizer.

Beyond the common arguments, the following optimizer parameters can be set as `kwargs`:

  - `stopval`
  - `xtol_rel`
  - `xtol_abs`
  - `constrtol_abs`
  - `initial_step`
  - `population`
  - `vector_storage`

## Local Optimizer

### Derivative-Free

Derivative-free optimizers are optimizers that can be used even in cases where no derivatives or automatic differentiation is specified. While they tend to be less efficient than derivative-based optimizers, they can be easily applied to cases where defining derivatives is difficult. Note that while these methods do not support general constraints, all support bounds constraints via `lb` and `ub` in the `OptimizationProblem`.

`NLopt` derivative-free optimizers are:

  - `NLopt.LN_PRAXIS()`
  - `NLopt.LN_COBYLA()`
  - `NLopt.LN_NEWUOA()`
  - `NLopt.LN_NEWUOA_BOUND()`
  - `NLopt.LN_NELDERMEAD()`
  - `NLopt.LN_SBPLX()`
  - `NLopt.LN_AUGLAG()`
  - `NLopt.LN_AUGLAG_EQ()`
  - `NLopt.LN_BOBYQA()`

The Rosenbrock function can be optimized using the `NLopt.LN_NELDERMEAD()` as follows:

```@example NLopt1
using Optimization
using OptimizationNLopt
rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p = [1.0, 100.0]
f = OptimizationFunction(rosenbrock)
prob = Optimization.OptimizationProblem(f, x0, p, lb = [-1.0, -1.0], ub = [1.0, 1.0])
sol = solve(prob, NLopt.LN_NELDERMEAD())
```

### Gradient-Based

Gradient-based optimizers are optimizers which utilize the gradient information based on derivatives defined or automatic differentiation.

`NLopt` gradient-based optimizers are:

  - `NLopt.LD_LBFGS_NOCEDAL()`
  - `NLopt.LD_LBFGS()`
  - `NLopt.LD_VAR1()`
  - `NLopt.LD_VAR2()`
  - `NLopt.LD_TNEWTON()`
  - `NLopt.LD_TNEWTON_RESTART()`
  - `NLopt.LD_TNEWTON_PRECOND()`
  - `NLopt.LD_TNEWTON_PRECOND_RESTART()`
  - `NLopt.LD_MMA()`
  - `NLopt.LD_AUGLAG()`
  - `NLopt.LD_AUGLAG_EQ()`
  - `NLopt.LD_SLSQP()`
  - `NLopt.LD_CCSAQ()`

The Rosenbrock function can be optimized using `NLopt.LD_LBFGS()` as follows:

```@example NLopt2
using Optimization, OptimizationNLopt
rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p = [1.0, 100.0]
f = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
prob = Optimization.OptimizationProblem(f, x0, p, lb = [-1.0, -1.0], ub = [1.0, 1.0])
sol = solve(prob, NLopt.LD_LBFGS())
```

## Global Optimizer

### Without Constraint Equations

The following algorithms in [`NLopt`](https://github.com/JuliaOpt/NLopt.jl) are performing global optimization on problems without
constraint equations. However, lower and upper constraints set by `lb` and `ub` in the `OptimizationProblem` are required.

`NLopt` global optimizers which fall into this category are:

  - `NLopt.GN_DIRECT()`
  - `NLopt.GN_DIRECT_L()`
  - `NLopt.GN_DIRECT_L_RAND()`
  - `NLopt.GN_DIRECT_NOSCAL()`
  - `NLopt.GN_DIRECT_L_NOSCAL()`
  - `NLopt.GN_DIRECT_L_RAND_NOSCAL()`
  - `NLopt.GD_STOGO()`
  - `NLopt.GD_STOGO_RAND()`
  - `NLopt.GN_CRS2_LM()`
  - `NLopt.GN_MLSL()`
  - `NLopt.GD_MLSL()`
  - `NLopt.GN_MLSL_LDS()`
  - `NLopt.GD_MLSL_LDS()`
  - `NLopt.G_MLSL()`
  - `NLopt.G_MLSL_LDS()`
  - `NLopt.GN_ESCH()`

The Rosenbrock function can be optimized using `NLopt.GN_DIRECT()` as follows:

```@example NLopt3
using Optimization, OptimizationNLopt
rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p = [1.0, 100.0]
f = OptimizationFunction(rosenbrock)
prob = Optimization.OptimizationProblem(f, x0, p, lb = [-1.0, -1.0], ub = [1.0, 1.0])
sol = solve(prob, NLopt.GN_DIRECT(), maxtime = 10.0)
```

Algorithms such as `NLopt.G_MLSL()` or `NLopt.G_MLSL_LDS()` also require a local optimizer to be selected,
which via the `local_method` argument of `solve`.

The Rosenbrock function can be optimized using `NLopt.G_MLSL_LDS()` with `NLopt.LN_NELDERMEAD()` as the local optimizer.
The local optimizer maximum iterations are set via `local_maxiters`:

```@example NLopt4
using Optimization, OptimizationNLopt
rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p = [1.0, 100.0]
f = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
prob = Optimization.OptimizationProblem(f, x0, p, lb = [-1.0, -1.0], ub = [1.0, 1.0])
sol = solve(prob, NLopt.G_MLSL_LDS(), local_method = NLopt.LD_LBFGS(), maxtime = 10.0,
    local_maxiters = 10)
```

### With Constraint Equations

The following algorithms in [`NLopt`](https://github.com/JuliaOpt/NLopt.jl) are performing global optimization on problems with
constraint equations. However, lower and upper constraints set by `lb` and `ub` in the `OptimizationProblem` are required.

!!! note "Constraints with NLopt"
    

Equality and inequality equation support for `NLopt` via `Optimization` is not supported directly. However, you can use the MOI wrapper to use constraints with NLopt optimizers.

`NLopt` global optimizers which fall into this category are:

  - `NLopt.GN_ORIG_DIRECT()`
  - `NLopt.GN_ORIG_DIRECT_L()`
  - `NLopt.GN_ISRES()`
  - `NLopt.GN_AGS()`
