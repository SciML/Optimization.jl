
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
