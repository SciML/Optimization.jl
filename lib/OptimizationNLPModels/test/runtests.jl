using OptimizationNLPModels, Optimization, NLPModelsTest, Ipopt, OptimizationMOI, Zygote,
      OptimizationOptimJL
using Test

@testset "NLPModels" begin
    # First problem: Problem 5 in the Hock-Schittkowski suite
    # Problem with box bounds
    hs5f(u, p) = sin(u[1] + u[2]) + (u[1] - u[2])^2 - (3 / 2) * u[1] + (5 / 2)u[2] + 1
    f = Optimization.OptimizationFunction(hs5f, Optimization.AutoZygote())
    lb = [-1.5; -3]
    ub = [4.0; 3.0]
    u0 = [0.0; 0.0]
    oprob = Optimization.OptimizationProblem(
        f, u0, lb = lb, ub = ub, sense = Optimization.MinSense)

    nlpmo = NLPModelsTest.HS5()
    converted = OptimizationNLPModels.OptimizationProblem(nlpmo, Optimization.AutoZygote())

    sol_native = solve(oprob, Ipopt.Optimizer())
    sol_converted = solve(converted, Ipopt.Optimizer())

    @test sol_converted.u ≈ sol_native.u

    # Second problem: Brown and Dennis function
    # Problem without bounds
    function brown_dennis(u, p)
        return sum([((u[1] + (i / 5) * u[2] - exp(i / 5))^2 +
                     (u[3] + sin(i / 5) * u[4] - cos(i / 5))^2)^2 for i in 1:20])
    end
    f = Optimization.OptimizationFunction(brown_dennis, Optimization.AutoZygote())
    u0 = [25.0; 5.0; -5.0; -1.0]
    oprob = Optimization.OptimizationProblem(f, u0, sense = Optimization.MinSense)

    nlpmo = NLPModelsTest.BROWNDEN()
    converted = OptimizationNLPModels.OptimizationProblem(nlpmo, Optimization.AutoZygote())

    sol_native = solve(oprob, BFGS())
    sol_converted = solve(converted, BFGS())

    @test sol_converted.u ≈ sol_native.u

    # Third problem: Problem 10 in the Hock-Schittkowski suite
    # Problem with inequality bounds
    hs10(u, p) = u[1] - u[2]
    hs10_cons(u, p) = -3.0 * u[1]^2 + 2.0 * u[1] * u[2] - u[2]^2 + 1.0
    lcons = [0.0]
    ucons = [Inf]
    u0 = [-10.0; 10.0]
    f = Optimization.OptimizationFunction(hs10, Optimization.AutoForwardDiff(); cons = hs10_cons)
    oprob = Optimization.OptimizationProblem(f, u0, lcons = lcons, ucons = ucons)

    nlpmo = NLPModelsTest.HS10()
    converted = OptimizationNLPModels.OptimizationProblem(nlpmo, Optimization.AutoForwardDiff())

    sol_native = solve(oprob, Ipopt.Optimizer())
    sol_converted = solve(converted, Ipopt.Optimizer())

    @test sol_converted.u ≈ sol_native.u
end
