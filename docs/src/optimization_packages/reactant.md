# [Reactant.jl](@id reactant)

## Installation: OptimizationReactant.jl

To use this package, install the OptimizationReactant package:

```julia
import Pkg;
Pkg.add("OptimizationReactant");
```

OptimizationReactant is an AD-backend sublibrary, not an optimizer: loading it
makes `adtype = AutoReactant()` available on `OptimizationFunction`s and
`OptimizationProblem`s. The objective, its gradient and the combined
value-and-gradient evaluation are compiled by
[Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) to StableHLO and
differentiated by [Enzyme](https://github.com/EnzymeAD/Enzyme.jl) inside the
compiled program — no Julia-level activity analysis runs on the objective.

```julia
using OptimizationBase, OptimizationReactant, OptimizationOptimisers
using Optimisers, Reactant

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
optf = OptimizationFunction(rosenbrock, AutoReactant())
prob = OptimizationProblem(optf, zeros(2), [1.0, 100.0])
sol = solve(prob, Optimisers.Adam(0.05); maxiters = 10_000)
```

The compiled program runs on whichever XLA client `Reactant` selects — CPU by
default, or the accelerator chosen through
`Reactant.XLA.set_default_backend("gpu")` before the first compile.

## Device placement

`AutoReactant` accepts host `Array` arguments: each argument signature gets
its own compiled program, and host inputs are copied to the device per call.
To avoid the per-iteration transfers, place `u0`/`p` on the device once:

```julia
prob_r = remake(prob; u0 = Reactant.to_rarray(prob.u0), p = Reactant.to_rarray(prob.p))
sol = solve(prob_r, Optimisers.Adam(0.05); maxiters = 10_000)
```

`p` is passed to the compiled program at run time, so mutating the parameter
buffers in place — e.g. `prob.p .= new_values` or `remake(prob; p = new_p)` —
is observed by the compiled objective and gradient; nothing is baked in.

## Caveats

  - First-order derivatives only: Hessians, Hessian-vector products, and
    constraint derivatives are not generated. Solvers that require them throw
    an `ArgumentError` at `init`; pass the functions explicitly to
    `OptimizationFunction` or choose a different `adtype`.
  - `AutoSparse{AutoReactant}` and `SecondOrder{<:AutoReactant}` are rejected.
  - The solver loop must be array-generic over the problem's array type:
    `OptimizationOptimisers` and `SimpleOptimization` work, including on
    `ConcreteRArray` state; solvers with non-generic inner loops (e.g. the
    Fortran-backed LBFGSB) do not.
  - Scalar `x[i]` indexing in the objective is allowed (it compiles to
    gather/slice ops) but vectorized expressions compile to faster code.
  - `AutoReactant(; mode = AutoEnzyme(...))` selects the Enzyme mode used
    inside the compiled program (default `Reverse`).
