using OptimizationAccelerated, Optimization
using ReverseDiff

function rosenbrock(x, p=nothing)
    (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
end

function con2_c(res, x, p)
    res .= [x[1]^2 + x[2]^2, x[2] * sin(x[1]) - x[1]]
end

x0 = zeros(2)
l1 = rosenbrock(x0, nothing)
optprob = OptimizationFunction(rosenbrock, Optimization.AutoReverseDiff(), cons = cons2_c)
prob = Optimization.OptimizationProblem(optprob, x0)

sol = solve(prob, OptimizationAccelerated.AcceleratedOpt(0.5, 0.5, 1.0, 0.5, 1e-6, 1e-6, 1e-6))
