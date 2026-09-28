# Multi-objective problems: weighted-sum scalarization with adaptive weights

A multi-objective `OptimizationProblem` — one whose objective `f(u, p)` returns a
vector with one entry per objective — can be solved with a single-objective
optimizer by minimizing a weighted sum of the objectives. [`weighted_sum`](@ref)
builds that scalarization, and the adaptive-weight rules update the weights
during the solve so that objectives of very different scales stay balanced.

## Scalarize the problem

```@example adaptive_weights
using Optimization, OptimizationOptimisers, Random, ForwardDiff

mof = MultiObjectiveOptimizationFunction(
    (u, p) -> [(u[1] - 1)^2, 3 * (u[1] + 1)^2], AutoForwardDiff()
)
prob = OptimizationProblem(mof, [0.0])
sprob = weighted_sum(prob)
```

`weighted_sum` produces an ordinary single-objective `OptimizationProblem` whose
objective is `weights ⋅ f(u, p)`; it shares a mutable weight vector with the
rules below. `weights` defaults to one entry of `1.0` per objective.

## Solve with an adaptive-weight rule

Each rule is built from the scalarized problem and passed as the `callback` of
`solve`; it reads `state.iter`, `state.u`, and `state.p`, and updates the
weights every `every` iterations. Any first-order optimizer that supports
callbacks can drive the solve:

```@example adaptive_weights
callback = ReLoBRaLo(sprob; every = 25, rng = Random.Xoshiro(0))
sol = solve(sprob, Adam(0.05); callback, maxiters = 100, save_best = false)
callback.weights
```

Use `save_best = false` with `OptimizationOptimisers` algorithms: the default
`save_best = true` keeps the iterate with the lowest *weighted* objective across
the run, but objective values under changing weights are not comparable, and on
the last iteration `save_best` reverts `θ` and re-invokes the callback at the
same iteration — an update the rules skip on purpose.

## Available rules

- [`GradientScale`](@ref) balances per-objective gradient magnitudes following
  Wang, Teng, and Perdikaris (2020). It needs per-objective gradients: pass
  `jac` to the `MultiObjectiveOptimizationFunction` or scalarize with a
  non-`NoAD` `adtype`.
- [`MiniMax`](@ref) ascends the weights with an `Optimisers.jl` rule, following
  McClenny and Braga-Neto (2020).
- [`SoftAdapt`](@ref) weights objectives by a softmax of their relative loss
  changes, following Heydari, Thompson, and Mehmood (2019).
- [`ReLoBRaLo`](@ref) combines random lookbacks to earlier losses with a moving
  average, following Bischof and Kraus (2021).

See each rule's docstring for its parameters and the cited paper.
