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
> supports linear, second-order-cone, and semidefinite problems, as the first
> vertical slice of the larger effort to make SymbolicAnalysis.jl + Optimization.jl a Convex.jl /
> cvxpy replacement (roadmap: [SciML/SymbolicAnalysis.jl#121](https://github.com/SciML/SymbolicAnalysis.jl/issues/121)).

`ConeConstraint(g, MOI.PositiveSemidefiniteConeTriangle(n))` accepts the upper
triangle of an affine symmetric matrix. Return entries by columns:
`[X[1,1], X[1,2], X[2,2], X[1,3], X[2,3], X[3,3], …]`. The matching
`sol.dual` entry uses the same order. Symmetric affine matrix atoms `eigmax`,
`eigmin`, and `logdet`, plus rectangular matrix `opnorm`, lower to semidefinite
cones; `logdet` uses MathOptInterface's log-determinant bridge with Clarabel.
