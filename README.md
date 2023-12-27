# Optimization.jl

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/Optimization/stable/)

[![codecov](https://codecov.io/gh/SciML/Optimization.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/SciML/Optimization.jl)
[![Build Status](https://github.com/SciML/Optimization.jl/workflows/CI/badge.svg)](https://github.com/SciML/Optimization.jl/actions/workflows/CI.yml?query=branch%3Amaster++)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7738525.svg)](https://doi.org/10.5281/zenodo.7738525)

Optimization.jl is a package with a scope that is beyond your normal global optimization
package. Optimization.jl seeks to bring together all of the optimization packages
it can find, local and global, into one unified Julia interface. This means, you
learn one package and you learn them all! Optimization.jl adds a few high-level
features, such as integrating with automatic differentiation, to make its usage
fairly simple for most cases, while allowing all of the options in a single
unified interface.

## Installation

Assuming that you already have Julia correctly installed, it suffices to import
Optimization.jl in the standard way:

```julia
using Pkg
Pkg.add("Optimization")
```

The packages relevant to the core functionality of Optimization.jl will be imported
accordingly and, in most cases, you do not have to worry about the manual
installation of dependencies. Below is the list of packages that need to be
installed explicitly if you intend to use the specific optimization algorithms
offered by them:

  - OptimizationBBO for [BlackBoxOptim.jl](https://github.com/robertfeldt/BlackBoxOptim.jl)
  - OptimizationEvolutionary for [Evolutionary.jl](https://github.com/wildart/Evolutionary.jl) (see also [this documentation](https://wildart.github.io/Evolutionary.jl/dev/))
  - OptimizationGCMAES for [GCMAES.jl](https://github.com/AStupidBear/GCMAES.jl)
  - OptimizationMOI for [MathOptInterface.jl](https://github.com/jump-dev/MathOptInterface.jl) (usage of algorithm via MathOptInterface API; see also the API [documentation](https://jump.dev/MathOptInterface.jl/stable/))
  - OptimizationMetaheuristics for [Metaheuristics.jl](https://github.com/jmejia8/Metaheuristics.jl) (see also [this documentation](https://jmejia8.github.io/Metaheuristics.jl/stable/))
  - OptimizationMultistartOptimization for [MultistartOptimization.jl](https://github.com/tpapp/MultistartOptimization.jl) (see also [this documentation](https://juliahub.com/docs/MultistartOptimization/cVZvi/0.1.0/))
  - OptimizationNLopt for [NLopt.jl](https://github.com/JuliaOpt/NLopt.jl) (usage via the NLopt API; see also the available [algorithms](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/))
  - OptimizationNOMAD for [NOMAD.jl](https://github.com/bbopt/NOMAD.jl) (see also [this documentation](https://bbopt.github.io/NOMAD.jl/stable/))
  - OptimizationNonconvex for [Nonconvex.jl](https://github.com/JuliaNonconvex/Nonconvex.jl) (see also [this documentation](https://julianonconvex.github.io/Nonconvex.jl/stable/))
  - OptimizationQuadDIRECT for [QuadDIRECT.jl](https://github.com/timholy/QuadDIRECT.jl)
  - OptimizationSpeedMapping for [SpeedMapping.jl](https://github.com/NicolasL-S/SpeedMapping.jl) (see also [this documentation](https://nicolasl-s.github.io/SpeedMapping.jl/stable/))

## Tutorials and Documentation

For information on using the package,
[see the stable documentation](https://docs.sciml.ai/Optimization/stable/). Use the
[in-development documentation](https://docs.sciml.ai/Optimization/dev/) for the version of
the documentation, which contains the unreleased features.

## Examples

```julia
using Optimization
rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p = [1.0, 100.0]

prob = OptimizationProblem(rosenbrock, x0, p)

using OptimizationOptimJL
sol = solve(prob, NelderMead())

using OptimizationBBO
prob = OptimizationProblem(rosenbrock, x0, p, lb = [-1.0, -1.0], ub = [1.0, 1.0])
sol = solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited())
```

Note that Optim.jl is a core dependency of Optimization.jl. However, BlackBoxOptim.jl
is not and must already be installed (see the list above).

*Warning:* The output of the second optimization task (`BBO_adaptive_de_rand_1_bin_radiuslimited()`) is
currently misleading in the sense that it returns `Status: failure (reached maximum number of iterations)`. However, convergence is actually
reached and the confusing message stems from the reliance on the Optim.jl output
struct (where the situation of reaching the maximum number of iterations is
rightly regarded as a failure). The improved output struct will soon be
implemented.

The output of the first optimization task (with the `NelderMead()` algorithm)
is given below:

```
* Status: success

* Candidate solution
   Final objective value:     3.525527e-09

* Found with
   Algorithm:     Nelder-Mead

* Convergence measures
   √(Σ(yᵢ-ȳ)²)/n ≤ 1.0e-08

* Work counters
   Seconds run:   0  (vs limit Inf)
   Iterations:    60
   f(x) calls:    118
```

We can also explore other methods in a similar way:

```julia
using ForwardDiff
f = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
prob = OptimizationProblem(f, x0, p)
sol = solve(prob, BFGS())
```

For instance, the above optimization task produces the following output:

```
* Status: success

* Candidate solution
   Final objective value:     7.645684e-21

* Found with
   Algorithm:     BFGS

* Convergence measures
   |x - x'|               = 3.48e-07 ≰ 0.0e+00
   |x - x'|/|x'|          = 3.48e-07 ≰ 0.0e+00
   |f(x) - f(x')|         = 6.91e-14 ≰ 0.0e+00
   |f(x) - f(x')|/|f(x')| = 9.03e+06 ≰ 0.0e+00
   |g(x)|                 = 2.32e-09 ≤ 1.0e-08

* Work counters
   Seconds run:   0  (vs limit Inf)
   Iterations:    16
   f(x) calls:    53
   ∇f(x) calls:   53
```

```julia
prob = OptimizationProblem(f, x0, p, lb = [-1.0, -1.0], ub = [1.0, 1.0])
sol = solve(prob, Fminbox(GradientDescent()))
```

The examples clearly demonstrate that Optimization.jl provides an intuitive
way of specifying optimization tasks and offers a relatively
easy access to a wide range of optimization algorithms.
