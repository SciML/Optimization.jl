
struct OptimizationStats
    iterations::Int
    time::Float64
    fevals::Int
    gevals::Int
    hevals::Int
end

OptimizationStats(; iterations = 0, time = 0.0, fevals = 0, gevals = 0, hevals = 0) =
    OptimizationStats(iterations, time, fevals, gevals, hevals)

struct OptimizationState{X, O, G, H, S}
    iteration::Int
    u::X
    objective::O
    gradient::G
    hessian::H
    solver_state::S
end

OptimizationState(; iterations = 0, u = nothing, objective = nothing,
        gradient = nothing, hessian = nothing, solver_state = nothing) =
    OptimizationState(iterations, u, objective, gradient, hessian, solver_state)
