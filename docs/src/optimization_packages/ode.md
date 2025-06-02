# OptimizationODE.jl

**OptimizationODE.jl** provides ODE-based optimization methods as a solver plugin for [SciML's Optimization.jl](https://github.com/SciML/Optimization.jl). It wraps various ODE solvers to perform gradient-based optimization using continuous-time dynamics.

## Installation

```julia
using Pkg
Pkg.add(url="OptimizationODE.jl")
```

## Usage

```julia
using OptimizationODE, Optimization, ADTypes, SciMLBase

function f(x, p)
    return sum(abs2, x)
end

function g!(g, x, p)
    @. g = 2 * x
end

x0 = [2.0, -3.0]
p = []

f_manual = OptimizationFunction(f, SciMLBase.NoAD(); grad = g!)
prob_manual = OptimizationProblem(f_manual, x0)

opt = ODEGradientDescent(dt=0.01)
sol = solve(prob_manual, opt; maxiters=50_000)

@show sol.u
@show sol.objective
```

## Available Optimizers

* `ODEGradientDescent(dt=...)` — uses the explicit Euler method.
* `RKChebyshevDescent()` — uses the ROCK2 method.
* `RKAccelerated()` — uses the Tsit5 Runge-Kutta method.
* `HighOrderDescent()` — uses the Vern7 high-order Runge-Kutta method.

## Interface Details

All optimizers require gradient information (either via automatic differentiation or manually provided `grad!`).

### Keyword Arguments

* `dt` — time step size (only for `ODEGradientDescent`).
* `maxiters` — maximum number of ODE steps.
* `callback` — function to observe progress.
* `progress=true` — enables live progress display.

## Development

Please refer to the `runtests.jl` file for a complete set of tests that demonstrate how each optimizer is used.

