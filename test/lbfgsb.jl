using Optimization
using ForwardDiff, Zygote, ReverseDiff, FiniteDiff, Tracker
using ModelingToolkit, Enzyme, Random

x0 = zeros(2)
rosenbrock(x, p = nothing) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
l1 = rosenbrock(x0)

optf = OptimizationFunction(rosenbrock, AutoForwardDiff())
prob = OptimizationProblem(optf, x0)
res = solve(prob, Optimization.LBFGS(), maxiters = 100)

@test res.u≈[1.0, 1.0] atol=1e-3

optf = OptimizationFunction(rosenbrock, AutoZygote())
prob = OptimizationProblem(optf, x0, lb = [0.0, 0.0], ub = [0.3, 0.3])
res = solve(prob, Optimization.LBFGS(), maxiters = 100)

@test res.u≈[0.3, 0.09] atol=1e-3
