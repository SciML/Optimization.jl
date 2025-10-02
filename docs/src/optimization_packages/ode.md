# OptimizationODE.jl

**OptimizationODE.jl** provides ODE-based optimization methods as a solver plugin for [SciML's Optimization.jl](https://github.com/SciML/Optimization.jl). It wraps various ODE solvers to perform gradient-based optimization using continuous-time dynamics.

## Installation

```julia
using Pkg
Pkg.add("OptimizationODE")
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

opt = ODEGradientDescent()
sol = solve(prob_manual, opt; dt=0.01, maxiters=50_000)

@show sol.u
@show sol.objective
```

## Local Gradient-based Optimizers

All provided optimizers are **gradient-based local optimizers** that solve optimization problems by integrating gradient-based ODEs to convergence:

* `ODEGradientDescent()` — performs basic gradient descent using the explicit Euler method. This is a simple and efficient method suitable for small-scale or well-conditioned problems.

* `RKChebyshevDescent()` — uses the ROCK2 solver, a stabilized explicit Runge-Kutta method suitable for stiff problems. It allows larger step sizes while maintaining stability.

* `RKAccelerated()` — leverages the Tsit5 method, a 5th-order Runge-Kutta solver that achieves faster convergence for smooth problems by improving integration accuracy.

* `HighOrderDescent()` — applies Vern7, a high-order (7th-order) explicit Runge-Kutta method for even more accurate integration. This can be beneficial for problems requiring high precision.

## DAE-based Optimizers

In addition to ODE-based optimizers, OptimizationODE.jl provides optimizers for differential-algebraic equation (DAE) constrained problems:

* `DAEMassMatrix()` — uses the Rodas5 solver (from OrdinaryDiffEq.jl) for DAE problems with a mass matrix formulation.

* `DAEIndexing()` — uses the IDA solver (from Sundials.jl) for DAE problems with index variable support.

You can also define a custom optimizer using the generic `ODEOptimizer(solver)` or `DAEOptimizer(solver)` constructor by supplying any ODE or DAE solver supported by [OrdinaryDiffEq.jl](https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/) or [Sundials.jl](https://github.com/SciML/Sundials.jl).

## DAE-based Optimizers

!!! warn
    DAE-based optimizers are still experimental and a research project. Use with caution.

In addition to ODE-based optimizers, OptimizationODE.jl provides optimizers for differential-algebraic equation (DAE) constrained problems:

* `DAEMassMatrix()` — uses the Rodas5P solver (from OrdinaryDiffEq.jl) for DAE problems with a mass matrix formulation.

* `DAEOptimizer(IDA())` — uses the IDA solver (from Sundials.jl) for DAE problems with index variable support (requires `using Sundials`)

You can also define a custom optimizer using the generic `ODEOptimizer(solver)` or `DAEOptimizer(solver)` constructor by supplying any ODE or DAE solver supported by [OrdinaryDiffEq.jl](https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/) or [Sundials.jl](https://github.com/SciML/Sundials.jl).

## Interface Details

All optimizers require gradient information (either via automatic differentiation or manually provided `grad!`). The optimization is performed by integrating the ODE defined by the negative gradient until a steady state is reached.
