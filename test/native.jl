using Optimization
using ForwardDiff, Zygote, ReverseDiff, FiniteDiff
using Test

x0 = zeros(2)
rosenbrock(x, p = nothing) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
l1 = rosenbrock(x0)

optf = OptimizationFunction(rosenbrock, AutoForwardDiff())
prob = OptimizationProblem(optf, x0)
@time res = solve(prob, Optimization.LBFGS(), maxiters = 100)
@test res.retcode == Optimization.SciMLBase.ReturnCode.Success

prob = OptimizationProblem(optf, x0, lb = [-1.0, -1.0], ub = [1.0, 1.0])
@time res = solve(prob, Optimization.LBFGS(), maxiters = 100)
@test res.retcode == Optimization.SciMLBase.ReturnCode.Success

function con2_c(res, x, p)
    res .= [x[1]^2 + x[2]^2, (x[2] * sin(x[1]) + x[1]) - 5]
end

optf = OptimizationFunction(rosenbrock, AutoZygote(), cons = con2_c)
prob = OptimizationProblem(optf, x0, lcons = [1.0, -Inf],
    ucons = [1.0, 0.0], lb = [-1.0, -1.0],
    ub = [1.0, 1.0])
@time res = solve(prob, Optimization.LBFGS(), maxiters = 100)
@test res.retcode == SciMLBase.ReturnCode.Success

using MLUtils, OptimizationOptimisers

x0 = (-pi):0.001:pi
y0 = sin.(x0)
data = MLUtils.DataLoader((x0, y0), batchsize = 126)
function loss(coeffs, data)
    ypred = [evalpoly(data[1][i], coeffs) for i in eachindex(data[1])]
    return sum(abs2, ypred .- data[2])
end

function cons1(res, coeffs, p = nothing)
    res[1] = coeffs[1] * coeffs[5] - 1
    return nothing
end

optf = OptimizationFunction(loss, AutoSparseForwardDiff(), cons = cons1)
callback = (st, l) -> (@show l; return false)

initpars = rand(5)
l0 = optf(initpars, (x0, y0))
prob = OptimizationProblem(optf, initpars, (x0, y0), lcons = [-Inf], ucons = [0.5],
    lb = [-10.0, -10.0, -10.0, -10.0, -10.0], ub = [10.0, 10.0, 10.0, 10.0, 10.0])
opt1 = solve(prob, Optimization.LBFGS(), maxiters = 1000, callback = callback)
@test opt1.objective < l0

prob = OptimizationProblem(optf, initpars, data, lcons = [-Inf], ucons = [1],
    lb = [-10.0, -10.0, -10.0, -10.0, -10.0], ub = [10.0, 10.0, 10.0, 10.0, 10.0])
opt = solve(
    prob, Optimization.AugLag(; inner = Adam()), maxiters = 10000, callback = callback)
@test opt.objective < l0

optf1 = OptimizationFunction(loss, AutoSparseForwardDiff())
prob1 = OptimizationProblem(optf1, rand(5), data)
sol1 = solve(prob1, OptimizationOptimisers.Adam(), maxiters = 1000, callback = callback)
@test sol1.objective < l0
