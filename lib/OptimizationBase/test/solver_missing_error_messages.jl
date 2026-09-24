using OptimizationBase, Test
using SciMLBase: NoAD

import OptimizationBase: allowscallback, requiresbounds, requiresconstraints

prob = OptimizationProblem((x, p) -> sum(x), zeros(2))
@test_throws OptimizationBase.OptimizerMissingError solve(prob, nothing)

struct OptAlgNoCb end

allowscallback(::OptAlgNoCb) = false
@test_throws OptimizationBase.IncompatibleOptimizerError solve(
    prob, OptAlgNoCb(),
    callback = (args...) -> false
)

struct OptAlgReqBounds end
requiresbounds(::OptAlgReqBounds) = true
@test_throws OptimizationBase.IncompatibleOptimizerError solve(prob, OptAlgReqBounds())

struct OptAlgDefault end

prob = OptimizationProblem((x, p) -> sum(x), zeros(2), lb = [-1.0, -1.0], ub = [1.0, 1.0])
@test_throws OptimizationBase.IncompatibleOptimizerError solve(prob, OptAlgDefault()) #by default allowsbounds is false

cons = (res, x, p) -> (res .= [x[1]^2 + x[2]^2])
optf = OptimizationFunction((x, p) -> sum(x), NoAD(), cons = cons)
prob = OptimizationProblem(optf, zeros(2))
@test_throws OptimizationBase.IncompatibleOptimizerError solve(prob, OptAlgDefault()) #by default allowsconstraints is false

struct OptAlgReqCons end
requiresconstraints(::OptAlgReqCons) = true
optf = OptimizationFunction((x, p) -> sum(x), NoAD())
prob = OptimizationProblem(optf, zeros(2))
@test_throws OptimizationBase.IncompatibleOptimizerError solve(prob, OptAlgReqCons())
