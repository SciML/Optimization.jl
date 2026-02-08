using OptimizationBase, ForwardDiff, SymbolicAnalysis, LinearAlgebra,
    Manifolds, OptimizationManopt, OptimizationLBFGSB

function f(x, p = nothing)
    return exp(x[1]) + x[1]^2
end

optf = OptimizationFunction(f, OptimizationBase.AutoForwardDiff())
prob = OptimizationProblem(optf, [0.4], structural_analysis = true)

@time sol = solve(prob, OptimizationLBFGSB.LBFGSB(), maxiters = 1000)
@test sol.cache.analysis_results.objective.curvature == SymbolicAnalysis.Convex
@test sol.cache.analysis_results.constraints === nothing

cvx_x0 = zeros(2)

optf = OptimizationFunction(rosenbrock, AutoEnzyme())
prob = OptimizationProblem(optf, cvx_x0, structural_analysis = true)
@time res = solve(prob, OptimizationLBFGSB.LBFGSB(), maxiters = 100)
@test res.cache.analysis_results.objective.curvature == SymbolicAnalysis.UnknownCurvature

function cvx_con2_c(res, x, p)
    return res .= [x[1]^2 + x[2]^2, (x[2] * sin(x[1]) + x[1]) - 5]
end

optf = OptimizationFunction(rosenbrock, AutoZygote(), cons = cvx_con2_c)
prob = OptimizationProblem(
    optf, cvx_x0, lcons = [1.0, -Inf], ucons = [1.0, 0.0],
    lb = [-1.0, -1.0], ub = [1.0, 1.0], structural_analysis = true
)
@time res = solve(prob, OptimizationLBFGSB.LBFGSB(), maxiters = 100)
@test res.cache.analysis_results.objective.curvature == SymbolicAnalysis.UnknownCurvature
@test res.cache.analysis_results.constraints[1].curvature == SymbolicAnalysis.Convex
@test res.cache.analysis_results.constraints[2].curvature ==
    SymbolicAnalysis.UnknownCurvature

# Manifold optimization test temporarily skipped due to Manopt linesearch issue
# producing NaN/Inf in SymmetricPositiveDefinite manifold optimization.
# See GitHub issue for tracking.
# m = 100
# σ = 0.005
# q = Matrix{Float64}(LinearAlgebra.I(5)) .+ 2.0
#
# M = SymmetricPositiveDefinite(5)
# data2 = [exp(M, q, σ * rand(M; vector_at = q)) for i in 1:m];
#
# f(x, p = nothing) = sum(SymbolicAnalysis.distance(M, data2[i], x)^2 for i in 1:5)
# optf = OptimizationFunction(f, OptimizationBase.AutoForwardDiff())
# prob = OptimizationProblem(optf, data2[1]; manifold = M, structural_analysis = true)
#
# opt = OptimizationManopt.GradientDescentOptimizer()
# @time sol = solve(prob, opt, maxiters = 100)
# @test sol.objective < 1.0e-1
# @test sol.cache.analysis_results.objective.curvature == SymbolicAnalysis.UnknownCurvature
# @test sol.cache.analysis_results.objective.gcurvature == SymbolicAnalysis.GConvex
