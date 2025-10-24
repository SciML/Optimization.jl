using OptimizationBase, Test
using SciMLBase: NoAD

import OptimizationBase: allowscallback, requiresbounds, requiresconstraints

prob = OptimizationProblem((x, p) -> sum(x), zeros(2))
@test_throws OptimizationBase.OptimizerMissingError solve(prob, nothing)

struct OptAlg end

allowscallback(::OptAlg) = false
@test_throws OptimizationBase.IncompatibleOptimizerError solve(prob, OptAlg(),
    callback = (args...) -> false)

requiresbounds(::OptAlg) = true
@test_throws OptimizationBase.IncompatibleOptimizerError solve(prob, OptAlg())
requiresbounds(::OptAlg) = false

prob = OptimizationProblem((x, p) -> sum(x), zeros(2), lb = [-1.0, -1.0], ub = [1.0, 1.0])
@test_throws OptimizationBase.IncompatibleOptimizerError solve(prob, OptAlg()) #by default allowsbounds is false

cons = (res, x, p) -> (res .= [x[1]^2 + x[2]^2])
optf = OptimizationFunction((x, p) -> sum(x), NoAD(), cons = cons)
prob = OptimizationProblem(optf, zeros(2))
@test_throws OptimizationBase.IncompatibleOptimizerError solve(prob, OptAlg()) #by default allowsconstraints is false

requiresconstraints(::OptAlg) = true
optf = OptimizationFunction((x, p) -> sum(x), NoAD())
prob = OptimizationProblem(optf, zeros(2))
@test_throws OptimizationBase.IncompatibleOptimizerError solve(prob, OptAlg())
