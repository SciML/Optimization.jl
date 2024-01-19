"""
$(TYPEDEF)

Stores the optimization run's state at the current iteration 
and is passed to the callback function as the first argument.

## Fields
- `iter`: current iteration
- `u`: current solution
- `objective`: current objective value
- `gradient`: current gradient
- `hessian`: current hessian
- `original`: if the solver has its own state object then it is stored here
"""
struct OptimizationState{X, O, G, H, S}
    iter::Int
    u::X
    objective::O
    grad::G
    hess::H
    original::S
end

function OptimizationState(; iter = 0, u = nothing, objective = nothing,
        grad = nothing, hess = nothing, original = nothing)
    OptimizationState(iter, u, objective, grad, hess, original)
end
