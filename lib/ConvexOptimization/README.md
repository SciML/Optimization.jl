# ConvexOptimization.jl

Disciplined-convex-programming backend for the SciML optimization stack.

`ConvexOptimization` solves a `SciMLBase.ConvexOptimizationProblem` by certifying
its convexity with [SymbolicAnalysis.jl](https://github.com/SciML/SymbolicAnalysis.jl),
lowering each atom to a [MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl)
cone, and calling a conic solver (default [Clarabel.jl](https://github.com/oxfordcontrol/Clarabel.jl)).
Unlike a general `OptimizationProblem` solved to a local optimum, a convex problem
is solved to a **global optimum**, and the returned `ConvexOptimizationSolution`
carries **dual multipliers** — the optimality certificate.

> **Status: experimental.** The current release targets the initial
> `ConvexOptimizationProblem`/`ConvexOptimizationSolution` interface
> ([SciML/SciMLBase.jl#1440](https://github.com/SciML/SciMLBase.jl/pull/1440)) and
> supports linear and second-order-cone problems, as the first vertical slice of
> the larger effort to make SymbolicAnalysis.jl + Optimization.jl a Convex.jl /
> cvxpy replacement (roadmap: [SciML/SymbolicAnalysis.jl#121](https://github.com/SciML/SymbolicAnalysis.jl/issues/121)).
