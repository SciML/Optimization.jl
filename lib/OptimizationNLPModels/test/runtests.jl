using OptimizationNLPModels, Optimization, NLPModelsTest
using Test

@testset "NLPModels" begin
    # Problem 5 in the Hock-Schittkowski suite
    hs5f(u, p) = sin(x[1] + x[2]) + (x[1] - x[2])^2 - (3 / 2) * x[1] + (5 / 2)x[2] + 1
    f = Optimization.OptimizationFunction((x, p) -> hs5f(u, p), Optimization.AutoZygote())
    lb = [-1.5; -3]
    ub = [4; 3]
    u0 = [0.0; 0.0]
    oprob = Optimization.OptimizationProblem(
        f, u0, lb = lb, ub = ub, sense = Optimization.MinSense)
    nlpmo = NLPModelsTest.HS5()

    converted = OptimizationNLPModels.OptimizationProblem(nlpmo)
end
