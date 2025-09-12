using Optimization, OptimizationBase, ForwardDiff, SymbolicAnalysis, LinearAlgebra,
      Manifolds, OptimizationManopt

function f(x, p = nothing)
    return exp(x[1]) + x[1]^2
end

optf = OptimizationFunction(f, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, [0.4], structural_analysis = true)

@time sol = solve(prob, Optimization.LBFGS(), maxiters = 1000)
@test sol.cache.analysis_results.objective.curvature == SymbolicAnalysis.Convex
@test sol.cache.analysis_results.constraints === nothing

x0 = zeros(2)
rosenbrock(x, p = nothing) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
l1 = rosenbrock(x0)

optf = OptimizationFunction(rosenbrock, AutoEnzyme())
prob = OptimizationProblem(optf, x0, structural_analysis = true)
@time res = solve(prob, Optimization.LBFGS(), maxiters = 100)
@test res.cache.analysis_results.objective.curvature == SymbolicAnalysis.UnknownCurvature

function con2_c(res, x, p)
    res .= [x[1]^2 + x[2]^2, (x[2] * sin(x[1]) + x[1]) - 5]
end

optf = OptimizationFunction(rosenbrock, AutoZygote(), cons = con2_c)
prob = OptimizationProblem(optf, x0, lcons = [1.0, -Inf], ucons = [1.0, 0.0],
    lb = [-1.0, -1.0], ub = [1.0, 1.0], structural_analysis = true)
@time res = solve(prob, Optimization.LBFGS(), maxiters = 100)
@test res.cache.analysis_results.objective.curvature == SymbolicAnalysis.UnknownCurvature
@test res.cache.analysis_results.constraints[1].curvature == SymbolicAnalysis.Convex
@test res.cache.analysis_results.constraints[2].curvature ==
      SymbolicAnalysis.UnknownCurvature

m = 100
σ = 0.005
q = Matrix{Float64}(LinearAlgebra.I(5)) .+ 2.0

M = SymmetricPositiveDefinite(5)
data2 = [exp(M, q, σ * rand(M; vector_at = q)) for i in 1:m];

f(x, p = nothing) = sum(SymbolicAnalysis.distance(M, data2[i], x)^2 for i in 1:5)
optf = OptimizationFunction(f, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, data2[1]; manifold = M, structural_analysis = true)

opt = OptimizationManopt.GradientDescentOptimizer()
@time sol = solve(prob, opt, maxiters = 100)
@test sol.minimum < 1e-1
@test sol.cache.analysis_results.objective.curvature == SymbolicAnalysis.UnknownCurvature
@test sol.cache.analysis_results.objective.gcurvature == SymbolicAnalysis.GConvex
