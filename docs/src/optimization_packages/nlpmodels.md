# NLPModels.jl

[NLPModels](https://jso.dev/NLPModels.jl/latest/), similarly to Optimization.jl itself,
provides a standardized modeling interface for representing Non-Linear Programs that
facilitates using different solvers on the same problem. The Optimization.jl extension of
NLPModels aims to provide a thin translation layer to make `NLPModel`s, the main export of
the package, compatible with the optimizers in the Optimization.jl ecosystem.

## Installation: NLPModels.jl

To translate an `NLPModel`, install the OptimizationNLPModels package:

```julia
import Pkg;
Pkg.add("OptimizationNLPModels")
```

The package NLPModels.jl itself contains no optimizers or models. Several packages
provide optimization problem ([CUTEst.jl](https://jso.dev/CUTEst.jl/stable/),
[NLPModelsTest.jl](https://jso.dev/NLPModelsTest.jl/dev/)) which can then be solved with
any optimizer supported by Optimization.jl

## Usage

For example, solving a problem defined in `NLPModelsTest` with
[`Ipopt.jl`](https://github.com/jump-dev/Ipopt.jl). First, install the packages like so:

```julia
import Pkg;
Pkg.add("NLPModelsTest", "Ipopt")
```

We instantiate [problem
10](https://jso.dev/NLPModelsTest.jl/dev/reference/#NLPModelsTest.HS10) in the
Hock--Schittkowski optimization suite available from `NLPModelsTest` as `HS10`, then
translate it to an `OptimizationProblem`.

```@example NLPModels
using OptimizationNLPModels, Optimization, NLPModelsTest, Ipopt
nlpmodel = NLPModelsTest.HS10()
prob = OptimizationProblem(nlpmodel, AutoForwardDiff())
```

which can now be solved like any other `OptimizationProblem`:

```@example NLPModels
sol = solve(prob, Ipopt.Optimizer())
```

## API

Problems represented as `NLPModel`s can be used to create `OptimizationProblem`s and
`OptimizationFunction`.

```@docs
SciMLBase.OptimizationFunction(::AbstractNLPModel, ::ADTypes.AbstractADType)
SciMLBase.OptimizationProblem(::AbstractNLPModel, ::ADTypes.AbstractADType)
```
