# [Reactant.jl](@id reactant)

## Installation: OptimizationReactant.jl

To use this package, install the OptimizationReactant package:

```julia
import Pkg;
Pkg.add("OptimizationReactant");
```

OptimizationReactant is an AD-backend sublibrary, not an optimizer: loading it
makes `adtype = AutoReactant()` available on `OptimizationFunction`s and
`OptimizationProblem`s. The objective and every requested derivative —
gradient, `fg`, Hessian, Hessian-vector product, `fgh`, and the constraint
Jacobian/VJP/JVP/Hessians and Lagrangian Hessian — are compiled by
[Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) to StableHLO and
differentiated by [Enzyme](https://github.com/EnzymeAD/Enzyme.jl) inside the
compiled program — no Julia-level activity analysis runs on the objective.
User-supplied derivative functions passed to `OptimizationFunction` are used
as-is.

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

  - `AutoSparse{AutoReactant}` and `SecondOrder{<:AutoReactant}` are rejected:
    `AutoReactant` already generates both derivative orders densely, and
    sparse detection cannot run through the compiled program.
  - Hessians are dense and assembled from `length(θ)` compiled
    Hessian-vector products, so second-order solvers are only practical for
    moderate parameter counts.
  - A bare `x[i] * x[j]` product inside the objective or constraints can be
    canonicalized into a `stablehlo.reduce` that Enzyme cannot
    differentiate; the program then fails to compile at `init`. Rewriting
    the product (e.g. a broadcast/vectorized expression) avoids it.
  - The solver loop must be array-generic over the problem's array type:
    `OptimizationOptimisers` and `SimpleOptimization` work, including on
    `ConcreteRArray` state; solvers with non-generic inner loops (e.g. the
    Fortran-backed LBFGSB) do not.
  - Scalar `x[i]` indexing in the objective is allowed (it compiles to
    gather/slice ops) but vectorized expressions compile to faster code.
  - `AutoReactant(; mode = AutoEnzyme(...))` selects the Enzyme mode used
    inside the compiled program (default `Reverse`).
