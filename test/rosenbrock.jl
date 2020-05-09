using GalacticOptim, Optim

rosenbrock(x,p) =  (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p  = [1.0,100.0]

prob = OptimizationProblem(rosenbrock,x0,p)
sol = solve(prob,BFGS())

rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
prob = OptimizationProblem(rosenbrock,p)
sol = solve(prob,BFGS())