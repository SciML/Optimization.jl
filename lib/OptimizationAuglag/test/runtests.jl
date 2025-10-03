using OptimizationBase
using MLUtils
using OptimizationOptimisers
using OptimizationAuglag
using ForwardDiff
using OptimizationBase: OptimizationCache
using SciMLBase: OptimizationFunction
using Test

@testset "OptimizationAuglag.jl" begin
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

    optf = OptimizationFunction(loss, OptimizationBase.AutoSparseForwardDiff(), cons = cons1)
    callback = (st, l) -> (@show l; return false)

    initpars = rand(5)
    l0 = optf(initpars, (x0, y0))

    prob = OptimizationProblem(optf, initpars, data, lcons = [-Inf], ucons = [1],
        lb = [-10.0, -10.0, -10.0, -10.0, -10.0], ub = [10.0, 10.0, 10.0, 10.0, 10.0])
    opt = solve(
        prob, OptimizationAuglag.AugLag(; inner = Adam()), maxiters = 10000, callback = callback)
    @test opt.objective < l0
end