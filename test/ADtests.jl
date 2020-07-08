using GalacticOptim, Optim, Test

x0 = zeros(2)
rosenbrock(x, p=nothing) =  (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
l1 = rosenbrock(x0)

optprob = OptimizationFunction(rosenbrock, x0, GalacticOptim.AutoForwardDiff())
prob = OptimizationProblem(optprob, x0)
sol = solve(prob, BFGS())
@test 10*sol.minimum < l1

prob = OptimizationProblem(optprob, x0)
sol = solve(prob, Newton())
@test 10*sol.minimum < l1

prob = OptimizationProblem(optprob, x0)
sol = solve(prob, Optim.KrylovTrustRegion())
@test 10*sol.minimum < l1

optprob = OptimizationFunction(rosenbrock, x0, GalacticOptim.AutoZygote())
prob = OptimizationProblem(optprob, x0)
sol = solve(prob, BFGS())
@test 10*sol.minimum < l1

prob = OptimizationProblem(optprob, x0)
sol = solve(prob, Newton())
@test 10*sol.minimum < l1

prob = OptimizationProblem(optprob, x0)
sol = solve(prob, Optim.KrylovTrustRegion())
@test 10*sol.minimum < l1