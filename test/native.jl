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

# Test Sophia with ComponentArrays + Enzyme (shadow generation fix)
using ComponentArrays
x0_comp = ComponentVector(a = 0.0, b = 0.0)
rosenbrock_comp(x, p = nothing) = (1 - x.a)^2 + 100 * (x.b - x.a^2)^2

optf_sophia = OptimizationFunction(rosenbrock_comp, AutoEnzyme())
prob_sophia = OptimizationProblem(optf_sophia, x0_comp)
res_sophia = solve(prob_sophia, Optimization.Sophia(η=0.01, k=5), maxiters = 50)
@test res_sophia.objective < rosenbrock_comp(x0_comp)  # Test optimization progress
@test res_sophia.retcode == Optimization.SciMLBase.ReturnCode.Success
