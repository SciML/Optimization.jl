# [Reexported API](@id reexports)

`using Optimization` (or `using OptimizationBase`) does not only bring in the names
Optimization.jl itself defines. It also re-surfaces a small, deliberately curated set
of names owned by other packages, so that stating and solving a problem needs a single
`using`. This page lists exactly what those names are and who owns them — the actual
documentation for each of them lives with its owner, linked below.

Each solver package documents its own re-exported algorithm names on its page under
"Optimizer Packages"; see for example [Optim.jl](@ref optim),
[Optimisers.jl](@ref optimisers), [NLopt.jl](@ref nlopt),
[Metaheuristics.jl](@ref metaheuristics), [NLPModels.jl](@ref nlpmodels) and
[Manopt.jl](@ref manopt).

## Reexported from SciMLBase

These names are owned and documented by
[SciMLBase.jl](https://docs.sciml.ai/SciMLBase/stable/):

  - Problems and functions: `OptimizationProblem`, `OptimizationFunction`,
    `MultiObjectiveOptimizationFunction`
  - Solutions: `OptimizationSolution`, `OptimizationStats`
  - Solving: `solve`, `init`, `solve!`, `remake`
  - Objective sense: `ObjSense`, `MinSense`, `MaxSense`
  - Return status: `ReturnCode`, `successful_retcode`
  - Multistart/ensembles: `EnsembleProblem`, `EnsembleSerial`, `EnsembleThreads`,
    `EnsembleDistributed`
  - Derivative declaration: `NoAD`
  - Algorithm capability traits: `allowsbounds`, `requiresbounds`, `allowsconstraints`,
    `requiresconstraints`, `allowscallback`, `requiresgradient`, `requireshessian`,
    `requiresconsjac`, `requiresconshess`

Anything else from SciMLBase must be imported from SciMLBase directly.

## Reexported from ADTypes

The automatic differentiation selectors are owned and documented by
[ADTypes.jl](https://docs.sciml.ai/ADTypes/stable/); see also
[the AD choice recommendations](@ref ad):

  - `AutoEnzyme`, `AutoFiniteDiff`, `AutoForwardDiff`, `AutoMooncake`,
    `AutoReverseDiff`, `AutoSparse`, `AutoSymbolics`, `AutoTracker`, `AutoZygote`

Anything else from ADTypes — the less common backends, the sparsity detectors and the
coloring algorithms — must be imported from ADTypes directly.

## Not reexported

`SciMLOperators`, `SymbolicIndexingInterface`, `SciMLLogging` and the rest of
Optimization.jl's dependencies are deliberately *not* re-exported. Blanket-reexporting
`SciMLBase` alone used to put 263 names into every `using Optimization` namespace,
including problem types for equation classes Optimization.jl cannot solve. Reach those
names through their owner, e.g. `SciMLBase.NullParameters`.
