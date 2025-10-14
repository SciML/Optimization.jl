using OptimizationMadNLP
using OptimizationBase
using Test
import Zygote

function objective(x, ::Any)
    return x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]
end
function constraints(res, x, ::Any)
    res .= [
        x[1] * x[2] * x[3] * x[4],
        x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2
    ]
end

x0 = [1.0, 5.0, 5.0, 1.0]
optfunc = OptimizationFunction(objective, AutoSparse(OptimizationBase.AutoZygote()), cons=constraints)
prob = OptimizationProblem(optfunc, x0; sense = OptimizationBase.MinSense,
        lb = [1.0, 1.0, 1.0, 1.0],
        ub = [5.0, 5.0, 5.0, 5.0],
        lcons = [25.0, 40.0],
        ucons = [Inf, 40.0])

cache = init(prob, MadNLPOptimizer())

sol = OptimizationBase.solve!(cache)

@test SciMLBase.successful_retcode(sol)
