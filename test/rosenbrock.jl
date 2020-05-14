using GalacticOptim, Optim, Test

rosenbrock(x,p) =  (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p  = [1.0,100.0]

l1 = rosenbrock(x0,p)
prob = OptimizationProblem(rosenbrock,x0,p)
sol = solve(prob,SimulatedAnnealing())
@test 10*sol.minimum < l1

rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

l1 = rosenbrock(p)
prob = OptimizationProblem(rosenbrock,p)
sol = solve(prob,SimulatedAnnealing()) 
@test 10*sol.minimum < l1


optprob = OptimizationFunction(rosenbrock)
prob = OptimizationProblem(optprob,p)
sol = solve(prob,BFGS())
@test 10*sol.minimum < l1

optprob = OptimizationFunction(rosenbrock)
prob = OptimizationProblem(optprob,p)
sol = solve(prob,Newton())
@test 10*sol.minimum < l1