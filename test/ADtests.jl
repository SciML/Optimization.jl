using GalacticOptim, GalacticOptimJL, GalacticFluxJL, Test
using ForwardDiff, Zygote, ReverseDiff, FiniteDiff, Tracker

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

optf = OptimizationFunction(rosenbrock, GalacticOptim.AutoForwardDiff())
optprob = GalacticOptim.instantiate_function(optf,x0,GalacticOptim.AutoForwardDiff(),nothing)
optprob.grad(G2, x0)
@test G1 == G2
optprob.hess(H2, x0)
@test H1 == H2

prob = OptimizationProblem(optprob, x0)

sol = solve(prob, Optim.BFGS())
@test 10*sol.minimum < l1

sol = solve(prob, Optim.Newton())
@test 10*sol.minimum < l1

sol = solve(prob, Optim.KrylovTrustRegion())
@test 10*sol.minimum < l1

optf = OptimizationFunction(rosenbrock, GalacticOptim.AutoZygote())
optprob = GalacticOptim.instantiate_function(optf,x0,GalacticOptim.AutoZygote(),nothing)
optprob.grad(G2, x0)
@test G1 == G2
optprob.hess(H2, x0)
@test H1 == H2

prob = OptimizationProblem(optprob, x0)

sol = solve(prob, Optim.BFGS())
@test 10*sol.minimum < l1

sol = solve(prob, Optim.Newton())
@test 10*sol.minimum < l1

sol = solve(prob, Optim.KrylovTrustRegion())
@test 10*sol.minimum < l1

optf = OptimizationFunction(rosenbrock, GalacticOptim.AutoReverseDiff())
optprob = GalacticOptim.instantiate_function(optf,x0,GalacticOptim.AutoReverseDiff(),nothing)
optprob.grad(G2, x0)
@test G1 == G2
optprob.hess(H2, x0)
@test H1 == H2

prob = OptimizationProblem(optprob, x0)
sol = solve(prob, Optim.BFGS())
@test 10*sol.minimum < l1

sol = solve(prob, Optim.Newton())
@test 10*sol.minimum < l1

sol = solve(prob, Optim.KrylovTrustRegion())
@test 10*sol.minimum < l1

optf = OptimizationFunction(rosenbrock, GalacticOptim.AutoTracker())
optprob = GalacticOptim.instantiate_function(optf,x0,GalacticOptim.AutoTracker(),nothing)
optprob.grad(G2, x0)
@test G1 == G2
@test_throws ErrorException optprob.hess(H2, x0)


prob = OptimizationProblem(optprob, x0)

sol = solve(prob, Optim.BFGS())
@test 10*sol.minimum < l1

@test_throws ErrorException solve(prob, Newton())

optf = OptimizationFunction(rosenbrock, GalacticOptim.AutoFiniteDiff())
optprob = GalacticOptim.instantiate_function(optf,x0,GalacticOptim.AutoFiniteDiff(),nothing)
optprob.grad(G2, x0)
@test G1 ≈ G2 rtol=1e-6
optprob.hess(H2, x0)
@test H1 ≈ H2 rtol=1e-6

prob = OptimizationProblem(optprob, x0)
sol = solve(prob, Optim.BFGS())
@test 10*sol.minimum < l1

sol = solve(prob, Optim.Newton())
@test 10*sol.minimum < l1

sol = solve(prob, Optim.KrylovTrustRegion())
@test sol.minimum < l1 #the loss doesn't go below 5e-1 here

sol = solve(prob, Flux.ADAM(0.1), maxiters = 1000)
@test 10*sol.minimum < l1
