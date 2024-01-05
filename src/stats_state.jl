"""
$(TYPEDEF)

Stores the optimization run's statistics that is returned 
in the `stats` field of the `OptimizationResult`. 

## Fields
- `iterations`: number of iterations
- `time`: time taken to run the solver
- `fevals`: number of function evaluations
- `gevals`: number of gradient evaluations
- `hevals`: number of hessian evaluations

Default values for all the field are set to 0 and hence even when 
you might expect non-zero values due to unavilability of the information 
from the solver it would be 0.
"""
struct OptimizationStats
    iterations::Int
    time::Float64
    fevals::Int
    gevals::Int
    hevals::Int
end

function OptimizationStats(; iterations = 0, time = 0.0, fevals = 0, gevals = 0, hevals = 0)
    OptimizationStats(iterations, time, fevals, gevals, hevals)
end

"""
$(TYPEDEF)

Stores the optimization run's state at the current iteration 
and is passed to the callback function as the first argument.

## Fields
- `iteration`: current iteration
- `u`: current solution
- `objective`: current objective value
- `gradient`: current gradient
- `hessian`: current hessian
- `solver_state`: if the solver has its own state object then it is stored here
"""
struct OptimizationState{X, O, G, H, S}
    iteration::Int
    u::X
    objective::O
    gradient::G
    hessian::H
    solver_state::S
end

function OptimizationState(; iteration = 0, u = nothing, objective = nothing,
        gradient = nothing, hessian = nothing, solver_state = nothing)
    OptimizationState(iteration, u, objective, gradient, hessian, solver_state)
end
