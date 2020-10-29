# GalacticOptim.jl

[![Build Status](https://travis-ci.com/SciML/GalacticOptim.jl.svg?branch=master)](https://travis-ci.com/SciML/GalacticOptim.jl)

GalacticOptim.jl is a package with a scope that is beyond your normal global optimization
package. GalacticOptim.jl seeks to bring together all of the optimization packages
it can find, local and global, into one unified Julia interface. This means, you
learn one package and you learn them all! GalacticOptim.jl adds a few high-level
features, such as integrating with automatic differentiation, to make its usage
fairly simple for most cases, while allowing all of the options in a single
unified interface.

#### Note: This package is still in development. The README is currently both an active documentation and a development roadmap.

## Installation

Assuming that you already have Julia correctly installed, it suffices to import
GalacticOptim.jl in the standard way:

```julia
import Pkg; Pkg.add("GalacticOptim")
```
The packages relevant to the core functionality of GalacticOptim.jl will be imported
accordingly and, in most cases, you do not have to worry about the manual
installation of dependencies. Below is the list of packages that need to be
installed explicitly if you intend to use the specific optimization algorithms
offered by them:

- [BlackBoxOptim.jl](https://github.com/robertfeldt/BlackBoxOptim.jl) (solver: `BBO()`)
- [NLopt.jl](https://github.com/JuliaOpt/NLopt.jl) (usage via the NLopt API;
see also the available [algorithms](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/))
- [MultistartOptimization.jl](https://github.com/tpapp/MultistartOptimization.jl)
(see also [this documentation](https://juliahub.com/docs/MultistartOptimization/cVZvi/0.1.0/))
- [QuadDIRECT.jl](https://github.com/timholy/QuadDIRECT.jl)
- [Evolutionary.jl](https://github.com/wildart/Evolutionary.jl) (see also [this documentation](https://wildart.github.io/Evolutionary.jl/dev/))
- [CMAEvolutionStrategy.jl](https://github.com/jbrea/CMAEvolutionStrategy.jl)

## Examples

```julia
 using GalacticOptim, Optim
 rosenbrock(x,p) =  (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
 x0 = zeros(2)
 p  = [1.0,100.0]

 prob = OptimizationProblem(rosenbrock,x0,p)
 sol = solve(prob,NelderMead())


 using BlackBoxOptim
 prob = OptimizationProblem(rosenbrock, x0, p, lb = [-1.0,-1.0], ub = [1.0,1.0])
 sol = solve(prob,BBO())
```

Note that Optim.jl is a core dependency of GalaticOptim.jl. However, BlackBoxOptim.jl
is not and must already be installed (see the list above).

*Warning:* The output of the second optimization task (`BBO()`) is
currently misleading in the sense that it returns `* Status: failure
(reached maximum number of iterations)`. However, convergence is actually
reached and the confusing message stems from the reliance on the Optim.jl output
 struct (where the situation of reaching the maximum number of iterations is
rightly regarded as a failure). The improved output struct will soon be
implemented.

The output of the first optimization task (with the `NelderMead()` algorithm)
is given below:

```julia
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
 f = OptimizationFunction(rosenbrock, GalacticOptim.AutoForwardDiff())
 prob = OptimizationProblem(f, x0, p)
 sol = solve(prob,BFGS())
```
For instance, the above optimization task produces the following output:

```julia
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
 prob = OptimizationProblem(f, x0, p, lb = [-1.0,-1.0], ub = [1.0,1.0])
 sol = solve(prob, Fminbox(GradientDescent()))
```
The examples clearly demonstrate that GalacticOptim.jl provides an intuitive
way of specifying optimization tasks and offers a relatively
easy access to a wide range of optimization algorithms.

### Automatic Differentiation Choices

While one can fully define all of the derivative functions associated with
nonlinear constrained optimization directly, in many cases it's easiest to just
rely on automatic differentiation to derive those functions. In GalacticOptim.jl,
you can provide as few functions as you want, or give a differentiation library
choice.

- `AutoForwardDiff()`
- `AutoReverseDiff(compile=false)`
- `AutoTracker()`
- `AutoZygote()`
- `AutoFiniteDiff()`
- `AutoModelingToolkit()`

### API Documentation

```julia
OptimizationFunction(f, AutoForwardDiff();
                     grad = nothing,
                     hes = nothing,
                     hv = nothing,
                     chunksize = 1)
```

```julia
OptimizationProblem(f, x, p = DiffEqBase.NullParameters(),;
                    lb = nothing,
                    ub = nothing)
```

```julia
solve(prob,alg;kwargs...)
```

Keyword arguments:

  - `maxiters` (the maximum number of iterations)
  - `abstol` (absolute tolerance)
  - `reltol` (relative tolerance)

Output Struct:
