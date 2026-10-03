# ConvexOptimization.jl

`ConvexOptimization` is a conic backend for `SciMLBase.ConvexOptimizationProblem`. `ConvexMOI` certifies the original problem with SymbolicAnalysis, lowers supported atoms to MathOptInterface cones, and solves with a MathOptInterface optimizer (Clarabel by default). A successful solve is globally optimal; the result is a `SciMLBase.OptimizationSolution`, and `sol.dual` contains one entry per user constraint in input order.

```julia
using ConvexOptimization
using LinearAlgebra
import MathOptInterface as MOI
import Clarabel

f = OptimizationFunction((u, p) -> norm(u .- [1.0, 1.0]))
constraints = [
    ConeConstraint((u, p) -> [sum(u) - 1.0], MOI.Zeros(1)),
    ConeConstraint((u, p) -> u, MOI.Nonnegatives(2)),
]
prob = ConvexOptimizationProblem(f, [0.5, 0.5]; constraints)
sol = solve(prob, ConvexMOI(Clarabel.Optimizer))

@show sol.u sol.objective sol.dual
```

The example projects `[1, 1]` onto the simplex: `sol.u` is `[0.5, 0.5]`, the objective is `sqrt(0.5)`, and `sol.dual` has two entries corresponding to the equality and nonnegativity constraints.

## Supported expressions

Objective atoms include `norm(w, p)` for `p = 1, 2, Inf`, scalar affine `abs(w)`, affine `max`/`maximum` and `min`/`minimum`, `sum(abs.(w))`, `maximum(abs.(w))`, `exp` and `log` of scalar affine expressions, sums of squares and positive-semidefinite quadratic forms. Symmetric affine matrices support `eigmax`, `eigmin`, and concave `logdet` (maximize or hypograph); real affine matrices of any shape support `opnorm`. Supported atoms can be nested when SymbolicAnalysis certifies the original composition by disciplined convex programming (DCP) rules.

The same atoms can appear in `<=` and `>=` constraints (`MOI.Nonpositives` and `MOI.Nonnegatives`): convex atoms must enter a `<=` row with a nonnegative coefficient, and concave atoms must enter a `>=` row with a nonnegative coefficient. Other cone constraints, including equalities, require affine components. Each atom is lowered through an epigraph or hypograph; internal cone constraints do not add entries to `sol.dual`.

`ConeConstraint(g, MOI.PositiveSemidefiniteConeTriangle(n))` accepts the upper triangle of an affine symmetric matrix in column order: `[X[1,1], X[1,2], X[2,2], X[1,3], X[2,3], X[3,3], …]`. The matching `sol.dual` entry uses the same order.

## Parameters and limits

The parameter-affine (DPP) path lets `reinit!(cache; p = new_p)` update numeric cone data and solve again without repeating symbolic tracing or convexity analysis. Parameter-dependent data must fit the backend's affine-in-parameters rules. Nested expressions are certified with parameters treated as sign-unknown, so they are accepted only when convex for every parameter value; invalid curvature or atom signs are rejected.

The backend supports only the atom forms and cone placements above. It rejects nonconvex or uncertified expressions, non-affine components in other cones, norms outside `p = 1, 2, Inf`, and parameter-dependent or non-PSD matrices in quadratic forms. `eigmax`, `eigmin`, and `logdet` require a symmetric affine matrix argument. In particular, `norm` must receive an array expression; writing it by hand as `sqrt(sum(w .^ 2))` is unsupported.

The backend is experimental and is part of the roadmap to make SymbolicAnalysis.jl and Optimization.jl a Convex.jl/cvxpy alternative: [SymbolicAnalysis.jl roadmap](https://github.com/SciML/SymbolicAnalysis.jl/issues/121).
