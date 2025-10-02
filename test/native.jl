using Optimization
using ForwardDiff, Zygote, ReverseDiff, FiniteDiff
using Test
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

optf1 = OptimizationFunction(loss, AutoSparseForwardDiff())
prob1 = OptimizationProblem(optf1, rand(5), data)
sol1 = solve(prob1, OptimizationOptimisers.Adam(), maxiters = 1000, callback = callback)
@test sol1.objective < l0
