"""
    OptimizationState(; iter=0, u=nothing, objective=nothing, grad=nothing,
        hess=nothing, original=nothing, p=nothing)

State passed to an optimization callback after a solver step.

# Fields

- `iter`: current iteration.
- `u`: current optimization variables.
- `objective`: current objective value.
- `grad`: current gradient, when available.
- `hess`: current Hessian, when available.
- `original`: solver-specific state object, when available.
- `p`: current optimization parameters.

# Examples

```julia
state = OptimizationState(; iter = 2, u = [1.0], objective = 0.5)
state.iter == 2
```
"""
struct OptimizationState{X, O, G, H, S, P}
    iter::Int
    u::X
    objective::O
    grad::G
    hess::H
    original::S
    p::P
end

function OptimizationState(;
        iter = 0, u = nothing, objective = nothing,
        grad = nothing, hess = nothing, original = nothing, p = nothing
    )
    return OptimizationState(iter, u, objective, grad, hess, original, p)
end
