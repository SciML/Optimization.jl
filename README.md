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

```julia
 f = OptimizationFunction(rosenbrock, GalacticOptim.AutoForwardDiff())
 prob = OptimizationProblem(f, x0, p)
 sol = solve(prob,BFGS())
```

```julia
 prob = OptimizationProblem(f, x0, p, lb = [-1.0,-1.0], ub = [1.0,1.0])
 sol = solve(prob, Fminbox(GradientDescent()))
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

  - `maxiters`
  - `abstol`
  - `reltol`

Output Struct:
