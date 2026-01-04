using OptimizationBase
using OptimizationBase: ReturnCode
using SciMLBase: OptimizationFunction, OptimizationProblem
using ForwardDiff, Zygote
using OptimizationLBFGSB
using MLUtils
using LBFGSB
using Test

@testset "OptimizationLBFGSB.jl" begin
    x0 = zeros(2)
    rosenbrock(x, p = nothing) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
    l1 = rosenbrock(x0)

    optf = OptimizationFunction(rosenbrock, OptimizationBase.AutoForwardDiff())
    prob = OptimizationProblem(optf, x0)
    @time res = solve(prob, OptimizationLBFGSB.LBFGSB(), maxiters = 100)
    @test res.retcode == ReturnCode.Success

    prob = OptimizationProblem(optf, x0, lb = [-1.0, -1.0], ub = [1.0, 1.0])
    @time res = solve(prob, OptimizationLBFGSB.LBFGSB(), maxiters = 100)
    @test res.retcode == ReturnCode.Success

    function con2_c(res, x, p)
        res .= [x[1]^2 + x[2]^2, (x[2] * sin(x[1]) + x[1]) - 5]
    end

    optf = OptimizationFunction(rosenbrock, OptimizationBase.AutoZygote(), cons = con2_c)
    prob = OptimizationProblem(
        optf, x0, lcons = [1.0, -Inf],
        ucons = [1.0, 0.0], lb = [-1.0, -1.0],
        ub = [1.0, 1.0]
    )
    @time res = solve(prob, OptimizationLBFGSB.LBFGSB(), maxiters = 100)
    @test res.retcode == SciMLBase.ReturnCode.Success

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
    prob = OptimizationProblem(
        optf, initpars, (x0, y0), lcons = [-Inf], ucons = [0.5],
        lb = [-10.0, -10.0, -10.0, -10.0, -10.0], ub = [10.0, 10.0, 10.0, 10.0, 10.0]
    )
    opt1 = solve(prob, OptimizationLBFGSB.LBFGSB(), maxiters = 1000, callback = callback)
    @test opt1.objective < l0

    # Test for issue #1094: LBFGSB should return Failure when encountering Inf/NaN
    # at bounds (e.g., due to function singularity)
    @testset "Inf/NaN detection at bounds (issue #1094)" begin
        # Function with singularity at α = -1 (log(0) = -Inf)
        ne = [47.79, 54.64, 60.68, 65.85, 70.1]
        nt = [49.01, 56.09, 62.38, 67.8, 72.29]

        function chi2_singular(alpha, p)
            n_th = (1 + alpha[1]) * nt
            total = 0.0
            for i in eachindex(ne)
                if ne[i] == 0.0
                    total += 2 * n_th[i]
                else
                    total += 2 * (n_th[i] - ne[i] + ne[i] * log(ne[i] / n_th[i]))
                end
            end
            return total
        end

        # With bounds including singularity at -1, should fail
        optf_singular = OptimizationFunction(chi2_singular, OptimizationBase.AutoForwardDiff())
        prob_singular = OptimizationProblem(optf_singular, [0.0]; lb = [-1.0], ub = [1.0])
        res_singular = solve(prob_singular, OptimizationLBFGSB.LBFGSB())
        @test res_singular.retcode == ReturnCode.Failure

        # With safe bounds (away from singularity), should succeed
        # The optimizer should find a minimum with a negative value of alpha
        prob_safe = OptimizationProblem(optf_singular, [0.0]; lb = [-0.9], ub = [1.0])
        res_safe = solve(prob_safe, OptimizationLBFGSB.LBFGSB())
        @test res_safe.retcode == ReturnCode.Success
        # The minimum should be negative (somewhere between -0.1 and 0)
        @test res_safe.u[1] < 0.0
        @test res_safe.u[1] > -0.5
    end
end
