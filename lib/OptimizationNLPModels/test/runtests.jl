using OptimizationNLPModels, OptimizationBase, NLPModelsTest, Ipopt, OptimizationMOI, Zygote,
      ReverseDiff, OptimizationLBFGSB, OptimizationOptimJL
using Test

@testset "NLPModels" begin
    # First problem: Problem 5 in the Hock-Schittkowski suite
    # https://jso.dev/NLPModelsTest.jl/dev/reference/#NLPModelsTest.HS5
    # Problem with box bounds
    hs5f(u, p) = sin(u[1] + u[2]) + (u[1] - u[2])^2 - (3 / 2) * u[1] + (5 / 2)u[2] + 1
    f = OptimizationBase.OptimizationFunction(hs5f, OptimizationBase.AutoZygote())
    lb = [-1.5; -3]
    ub = [4.0; 3.0]
    u0 = [0.0; 0.0]
    oprob = OptimizationBase.OptimizationProblem(
        f, u0, lb = lb, ub = ub, sense = OptimizationBase.MinSense)

    nlpmo = NLPModelsTest.HS5()
    converted = OptimizationNLPModels.OptimizationProblem(nlpmo, OptimizationBase.AutoZygote())

    sol_native = solve(oprob, OptimizationLBFGSB.LBFGSB(), maxiters = 1000)
    sol_converted = solve(converted, OptimizationLBFGSB.LBFGSB(), maxiters = 1000)

    @test sol_converted.retcode == sol_native.retcode
    @test sol_converted.u ≈ sol_native.u

    # Second problem: Brown and Dennis function
    # https://jso.dev/NLPModelsTest.jl/dev/reference/#NLPModelsTest.BROWNDEN
    # Problem without bounds
    function brown_dennis(u, p)
        return sum([((u[1] + (i / 5) * u[2] - exp(i / 5))^2 + (u[3] + sin(i / 5) * u[4] - cos(i / 5))^2)^2 for i in 1:20])
    end
    f = OptimizationBase.OptimizationFunction(brown_dennis, OptimizationBase.AutoZygote())
    u0 = [25.0; 5.0; -5.0; -1.0]
    oprob = OptimizationBase.OptimizationProblem(f, u0, sense = OptimizationBase.MinSense)

    nlpmo = NLPModelsTest.BROWNDEN()
    converted = OptimizationNLPModels.OptimizationProblem(nlpmo, OptimizationBase.AutoZygote())

    sol_native = solve(oprob, BFGS())
    sol_converted = solve(converted, BFGS())

    @test sol_converted.retcode == sol_native.retcode
    @test sol_converted.u ≈ sol_native.u

    # Third problem: Problem 10 in the Hock-Schittkowski suite
    # https://jso.dev/NLPModelsTest.jl/dev/reference/#NLPModelsTest.HS10
    # Problem with inequality bounds
    hs10(u, p) = u[1] - u[2]
    hs10_cons(res, u, p) = (res .= -3.0 * u[1]^2 + 2.0 * u[1] * u[2] - u[2]^2 + 1.0)
    lcons = [0.0]
    ucons = [Inf]
    u0 = [-10.0; 10.0]
    f = OptimizationBase.OptimizationFunction(
        hs10, OptimizationBase.AutoForwardDiff(); cons = hs10_cons)
    oprob = OptimizationBase.OptimizationProblem(
        f, u0, lcons = lcons, ucons = ucons, sense = OptimizationBase.MinSense)

    nlpmo = NLPModelsTest.HS10()
    converted = OptimizationNLPModels.OptimizationProblem(
        nlpmo, OptimizationBase.AutoForwardDiff())

    sol_native = solve(oprob, Ipopt.Optimizer())
    sol_converted = solve(converted, Ipopt.Optimizer())

    @test sol_converted.retcode == sol_native.retcode
    @test sol_converted.u ≈ sol_native.u

    # Fourth problem: Problem 13 in the Hock-Schittkowski suite
    # https://jso.dev/NLPModelsTest.jl/dev/reference/#NLPModelsTest.HS13
    # Problem with box & inequality bounds
    hs13(u, p) = (u[1] - 2.0)^2 + u[2]^2
    hs13_cons(res, u, p) = (res .= (1.0 - u[1])^3 - u[2])
    lcons = [0.0]
    ucons = [Inf]
    lb = [0.0; 0.0]
    ub = [Inf; Inf]
    u0 = [-2.0; -2.0]
    f = OptimizationBase.OptimizationFunction(
        hs13, OptimizationBase.AutoForwardDiff(); cons = hs13_cons)
    oprob = OptimizationBase.OptimizationProblem(f, u0, lb = lb, ub = ub, lcons = lcons,
        ucons = ucons, sense = OptimizationBase.MinSense)

    nlpmo = NLPModelsTest.HS13()
    converted = OptimizationNLPModels.OptimizationProblem(
        nlpmo, OptimizationBase.AutoForwardDiff())

    sol_native = solve(oprob, Ipopt.Optimizer())
    sol_converted = solve(converted, Ipopt.Optimizer())

    @test sol_converted.retcode == sol_native.retcode
    @test sol_converted.u ≈ sol_native.u

    # Fifth problem: Problem 14 in the Hock-Schittkowski suite
    # https://jso.dev/NLPModelsTest.jl/dev/reference/#NLPModelsTest.HS14
    # Problem with mixed equality & inequality constraints
    hs14(u, p) = (u[1] - 2.0)^2 + (u[2] - 1.0)^2
    hs14_cons(res, u, p) = (res .= [u[1] - 2.0 * u[2];
                                    -0.25 * u[1]^2 - u[2]^2 + 1.0])
    lcons = [-1.0; 0.0]
    ucons = [-1.0; Inf]
    u0 = [2.0; 2.0]
    f = OptimizationBase.OptimizationFunction(
        hs14, OptimizationBase.AutoForwardDiff(); cons = hs14_cons)
    oprob = OptimizationBase.OptimizationProblem(
        f, u0, lcons = lcons, ucons = ucons, sense = OptimizationBase.MinSense)

    nlpmo = NLPModelsTest.HS14()
    converted = OptimizationNLPModels.OptimizationProblem(
        nlpmo, OptimizationBase.AutoForwardDiff())

    sol_native = solve(oprob, Ipopt.Optimizer())
    sol_converted = solve(converted, Ipopt.Optimizer())

    @test sol_converted.retcode == sol_native.retcode
    @test sol_converted.u ≈ sol_native.u
end
