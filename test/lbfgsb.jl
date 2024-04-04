using Optimization
using ForwardDiff, Zygote, ReverseDiff, FiniteDiff, Tracker
using ModelingToolkit, Enzyme, Random
using Test

x0 = zeros(2)
rosenbrock(x, p = nothing) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
l1 = rosenbrock(x0)

optf = OptimizationFunction(rosenbrock, AutoEnzyme())
prob = OptimizationProblem(optf, x0)
@time res = solve(prob, Optimization.LBFGS(), maxiters = 100)

@test res.u≈[1.0, 1.0] atol=1e-3

optf = OptimizationFunction(rosenbrock, AutoZygote())
prob = OptimizationProblem(optf, x0, lb = [0.0, 0.0], ub = [0.3, 0.3])
res = solve(prob, Optimization.LBFGS(), maxiters = 100)

@test res.u≈[0.3, 0.09] atol=1e-3

function con2_c(res, x, p)
    res .= [x[1]^2 + x[2]^2, (x[2] * sin(x[1]) + x[1])-5]
end

optf = OptimizationFunction(rosenbrock, AutoForwardDiff(), cons = con2_c)
prob = OptimizationProblem(optf, x0, lcons = [1.0, -Inf], ucons = [1.0, 0.0], lb = [-1.0, -1.0], ub = [1.0, 1.0])
res = solve(prob, Optimization.LBFGS(), maxiters = 100)

@test res.objective < l1 
cons_cache = [0.,0.]
con2_c(cons_cache, res.u, res.cache.p)
@test cons_cache[1] ≈ 1.0 atol=1e-3
@test cons_cache[2] < 0.0

optf = OptimizationFunction(rosenbrock, AutoZygote(), cons = con2_c)
prob = OptimizationProblem(optf, x0, lcons = [1.0, -Inf], ucons = [1.0, 0.0], lb = [-1.0, -1.0], ub = [1.0, 1.0])
res = solve(prob, Optimization.LBFGS(), maxiters = 100)

@test res.objective < l1