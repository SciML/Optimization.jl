using GalacticOptim, Optim, Test, ReverseDiff

x0 = zeros(2)
rosenbrock(x, p=nothing) =  (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
l1 = rosenbrock(x0)

function g!(G, x)
    G[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    G[2] = 200.0 * (x[2] - x[1]^2)
end

function h!(H, x)
    H[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
    H[1, 2] = -400.0 * x[1]
    H[2, 1] = -400.0 * x[1]
    H[2, 2] = 200.0
end

G1 = Array{Float64}(undef,2)
G2 = Array{Float64}(undef,2)
H1 = Array{Float64}(undef, 2, 2)
H2 = Array{Float64}(undef, 2, 2)

g!(G1, x0)
h!(H1, x0)

optprob = OptimizationFunction(rosenbrock, x0, GalacticOptim.AutoForwardDiff())
optprob.grad(G2, x0)
@test G1 == G2
optprob.hess(H2, x0)
@test H1 == H2

prob = OptimizationProblem(optprob, x0)

sol = solve(prob, BFGS())
@test 10*sol.minimum < l1

sol = solve(prob, Newton())
@test 10*sol.minimum < l1

sol = solve(prob, Optim.KrylovTrustRegion())
@test 10*sol.minimum < l1

optprob = OptimizationFunction(rosenbrock, x0, GalacticOptim.AutoZygote())
optprob.grad(G2, x0)
@test G1 == G2
optprob.hess(H2, x0)
@test H1 == H2

prob = OptimizationProblem(optprob, x0)
sol = solve(prob, BFGS())
@test 10*sol.minimum < l1

prob = OptimizationProblem(optprob, x0)
sol = solve(prob, Newton())
@test 10*sol.minimum < l1

prob = OptimizationProblem(optprob, x0)
sol = solve(prob, Optim.KrylovTrustRegion())
@test 10*sol.minimum < l1

optprob = OptimizationFunction(rosenbrock, x0, GalacticOptim.AutoReverseDiff())
optprob.grad(G2, x0)
@test G1 == G2
optprob.hess(H2, x0)
@test H1 == H2

prob = OptimizationProblem(optprob, x0)
sol = solve(prob, BFGS())
@test 10*sol.minimum < l1

prob = OptimizationProblem(optprob, x0)
sol = solve(prob, Newton())
@test 10*sol.minimum < l1

prob = OptimizationProblem(optprob, x0)
sol = solve(prob, Optim.KrylovTrustRegion())
@test 10*sol.minimum < l1