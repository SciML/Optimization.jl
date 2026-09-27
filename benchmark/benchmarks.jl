using Optimization, OptimizationOptimJL, ForwardDiff, BenchmarkTools

const SUITE = BenchmarkGroup()

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p = (1.0, 100.0)

# =============================================================================
# Problem construction
# =============================================================================

SUITE["construct"] = BenchmarkGroup()

SUITE["construct"]["optfunction_ad"] = @benchmarkable OptimizationFunction(
    $rosenbrock, Optimization.AutoForwardDiff()
)
SUITE["construct"]["optproblem"] = @benchmarkable OptimizationProblem(
    $(OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())), $x0, $p
)

optf = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, x0, p)

# =============================================================================
# Solves
# =============================================================================

SUITE["solve"] = BenchmarkGroup()

SUITE["solve"]["BFGS"] = @benchmarkable solve($prob, BFGS())
SUITE["solve"]["LBFGS"] = @benchmarkable solve($prob, LBFGS())
SUITE["solve"]["NelderMead"] = @benchmarkable solve(
    $(OptimizationProblem(rosenbrock, x0, p)), NelderMead()
)

# =============================================================================
# Larger problem
# =============================================================================

SUITE["larger"] = BenchmarkGroup()

rosenbrock_n(x, p) = sum(i -> (1.0 - x[i])^2 + 100.0 * (x[i + 1] - x[i]^2)^2, 1:99)
optf_n = OptimizationFunction(rosenbrock_n, Optimization.AutoForwardDiff())
prob_n = OptimizationProblem(optf_n, zeros(100), nothing)

SUITE["larger"]["BFGS_100d"] = @benchmarkable solve(
    $prob_n, BFGS(); maxiters = 100
)
