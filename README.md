# GalacticOptim.jl

[![Build Status](https://travis-ci.com/SciML/GalacticOptim.jl.svg?branch=master)](https://travis-ci.com/SciML/GalacticOptim.jl)

GalacticOptim.jl is a package with a scope that is beyond your normal global optimization
package. GalacticOptim.jl seeks to bring together all of the optimization packages
it can find, local and global, into one unified Julia interface. This means, you
learn one package and you learn them all! GalacticOptim.jl adds a few high level
features, such as integrating with automatic differentiation, to make its usage
fairly simple for most cases, while allowing all of the options in a single
unified interface.

#### Note: This package is currently in development and is not released. The README is currently a development roadmap.

## Examples

```julia
using GalacticOptim
rosenbrock(x,p) =  (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p  = [1.0,100.0]

prob = OptimizationProblem(f,x0,p)
sol = solve(prob,BFGS())

prob = OptimizationProblem(f,lower_bounds=[-1.0,-1.0],upper_bounds=[1.0,1.0])
sol = solve(prob,BFGS())

using BlackBoxOptim
sol = solve(prob,BBO())

using Flux
sol = solve(prob,ADAM(0.01),maxiters = 100)
```

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

### Symbolic DSL Interface

Provided by ModelingToolkit.jl

### API Documentation

```julia
OptimizationFunction(f;
                     grad = AutoForwardDiff(),
                     hes = AutoForwardDiff(),
                     eqconstraints = AutoForwardDiff(),
                     neqconstraints = AutoForwardDiff(),
                     eqconstraints_jac = AutoForwardDiff(),
                     neqconstraints_jac = AutoForwardDiff(),
                     colorvec,hessparsity,eqsparsity,neqsparsity)
```

```julia
OptimizationProblem(f,x0=nothing,p=nothing;
                    lower_bounds=nothing,
                    upper_bounds=nothing)
```

```julia
solve(prob,alg;kwargs...)
```

Keyword arguments:

  - `maxiters`
  - `abstol`
  - `reltol`

Output Struct:
