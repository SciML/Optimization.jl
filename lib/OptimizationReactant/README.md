# OptimizationReactant

[Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) backend for
[Optimization.jl](https://github.com/SciML/Optimization.jl). Loading this
package makes `adtype = AutoReactant()` available on
`OptimizationFunction`s/`OptimizationProblem`s: the objective, its gradient
and the combined value-and-gradient evaluation are compiled to StableHLO and
differentiated by Enzyme inside the compiled program.

```julia
using OptimizationBase, OptimizationReactant, OptimizationOptimisers
using Optimisers, Reactant

optf = OptimizationFunction((u, p) -> sum(abs2, u .- p), AutoReactant())
prob = OptimizationProblem(optf, zeros(3), ones(3))
sol = solve(prob, Optimisers.Adam(0.1); maxiters = 100)

# Device placement: move the problem data once with `to_rarray`
# (CPU PJRT by default, or the XLA client selected via
# `Reactant.XLA.set_default_backend("gpu")`).
prob_r = remake(prob; u0 = Reactant.to_rarray(prob.u0), p = Reactant.to_rarray(prob.p))
sol_r = solve(prob_r, Optimisers.Adam(0.1); maxiters = 100)
```

Host `Array` arguments work as well — each argument signature compiles its
own program on first use — but placing `u0`/`p` on the device avoids the
per-iteration transfers.

Hessians, Hessian-vector products, constraint derivatives and sparse
differentiation are not generated yet; solvers that require them throw an
`ArgumentError` at `init`. Array-generic first-order solvers
(`OptimizationOptimisers`, `SimpleOptimization`) are the intended consumers;
solvers with non-generic inner loops (e.g. the Fortran-backed LBFGSB) are not
compatible.
