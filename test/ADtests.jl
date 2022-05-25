using GalacticOptim, GalacticOptimJL, GalacticOptimisers, Test
using ForwardDiff, Zygote, ReverseDiff, FiniteDiff, Tracker
using ModelingToolkit
x0 = zeros(2)
rosenbrock(x, p=nothing) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
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

G1 = Array{Float64}(undef, 2)
G2 = Array{Float64}(undef, 2)
H1 = Array{Float64}(undef, 2, 2)
H2 = Array{Float64}(undef, 2, 2)

g!(G1, x0)
h!(H1, x0)

cons = (x, p) -> [x[1]^2 + x[2]^2]
optf = OptimizationFunction(rosenbrock, GalacticOptim.AutoModelingToolkit(), cons=cons)
optprob = GalacticOptim.instantiate_function(optf, x0, GalacticOptim.AutoModelingToolkit(), nothing, 1)
optprob.grad(G2, x0)
@test G1 == G2
optprob.hess(H2, x0)
@test H1 == H2
@test optprob.cons(x0) == [0.0]
J = Array{Float64}(undef, 2)
optprob.cons_j(J, [5.0, 3.0])
@test J == [10.0, 6.0]
H3 = [Array{Float64}(undef, 2, 2)]
optprob.cons_h(H3, x0)
@test H3 == [[2.0 0.0; 0.0 2.0]]

function con2_c(x, p)
    [x[1]^2 + x[2]^2, x[2] * sin(x[1]) - x[1]]
end
optf = OptimizationFunction(rosenbrock, GalacticOptim.AutoModelingToolkit(), cons=con2_c)
optprob = GalacticOptim.instantiate_function(optf, x0, GalacticOptim.AutoModelingToolkit(), nothing, 2)
optprob.grad(G2, x0)
@test G1 == G2
optprob.hess(H2, x0)
@test H1 == H2
@test optprob.cons(x0) == [0.0, 0.0]
J = Array{Float64}(undef, 2, 2)
optprob.cons_j(J, [5.0, 3.0])
@test all(isapprox(J, [10.0 6.0; -0.149013 -0.958924]; rtol=1e-3))
H3 = [Array{Float64}(undef, 2, 2), Array{Float64}(undef, 2, 2)]
optprob.cons_h(H3, x0)
@test H3 == [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

optf = OptimizationFunction(rosenbrock, GalacticOptim.AutoModelingToolkit(true, true), cons=con2_c)
optprob = GalacticOptim.instantiate_function(optf, x0, GalacticOptim.AutoModelingToolkit(true, true), nothing, 2)
using SparseArrays
sH = sparse([1, 1, 2, 2], [1, 2, 1, 2], zeros(4))
@test findnz(sH)[1:2] == findnz(optprob.hess_prototype)[1:2]
optprob.hess(sH, x0)
@test sH == H2
@test optprob.cons(x0) == [0.0, 0.0]
sJ = sparse([1, 1, 2, 2], [1, 2, 1, 2], zeros(4))
@test findnz(sJ)[1:2] == findnz(optprob.cons_jac_prototype)[1:2]
optprob.cons_j(sJ, [5.0, 3.0])
@test all(isapprox(sJ, [10.0 6.0; -0.149013 -0.958924]; rtol=1e-3))
sH3 = [sparse([1, 2], [1, 2], zeros(2)), sparse([1, 1, 2], [1, 2, 1], zeros(3))]
@test getindex.(findnz.(sH3), Ref([1,2])) == getindex.(findnz.(optprob.cons_hess_prototype), Ref([1,2]))
optprob.cons_h(sH3, x0)
@test Array.(sH3) == [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

optf = OptimizationFunction(rosenbrock, GalacticOptim.AutoForwardDiff())
optprob = GalacticOptim.instantiate_function(optf, x0, GalacticOptim.AutoForwardDiff(), nothing)
optprob.grad(G2, x0)
@test G1 == G2
optprob.hess(H2, x0)
@test H1 == H2

prob = OptimizationProblem(optprob, x0)

sol = solve(prob, Optim.BFGS())
@test 10 * sol.minimum < l1

sol = solve(prob, Optim.Newton())
@test 10 * sol.minimum < l1

sol = solve(prob, Optim.KrylovTrustRegion())
@test 10 * sol.minimum < l1

optf = OptimizationFunction(rosenbrock, GalacticOptim.AutoZygote())
optprob = GalacticOptim.instantiate_function(optf, x0, GalacticOptim.AutoZygote(), nothing)
optprob.grad(G2, x0)
@test G1 == G2
optprob.hess(H2, x0)
@test H1 == H2

prob = OptimizationProblem(optprob, x0)

sol = solve(prob, Optim.BFGS())
@test 10 * sol.minimum < l1

sol = solve(prob, Optim.Newton())
@test 10 * sol.minimum < l1

sol = solve(prob, Optim.KrylovTrustRegion())
@test 10 * sol.minimum < l1

optf = OptimizationFunction(rosenbrock, GalacticOptim.AutoReverseDiff())
optprob = GalacticOptim.instantiate_function(optf, x0, GalacticOptim.AutoReverseDiff(), nothing)
optprob.grad(G2, x0)
@test G1 == G2
optprob.hess(H2, x0)
@test H1 == H2

prob = OptimizationProblem(optprob, x0)
sol = solve(prob, Optim.BFGS())
@test 10 * sol.minimum < l1

sol = solve(prob, Optim.Newton())
@test 10 * sol.minimum < l1

sol = solve(prob, Optim.KrylovTrustRegion())
@test 10 * sol.minimum < l1

optf = OptimizationFunction(rosenbrock, GalacticOptim.AutoTracker())
optprob = GalacticOptim.instantiate_function(optf, x0, GalacticOptim.AutoTracker(), nothing)
optprob.grad(G2, x0)
@test G1 == G2
@test_throws ErrorException optprob.hess(H2, x0)


prob = OptimizationProblem(optprob, x0)

sol = solve(prob, Optim.BFGS())
@test 10 * sol.minimum < l1

@test_throws ErrorException solve(prob, Newton())

optf = OptimizationFunction(rosenbrock, GalacticOptim.AutoFiniteDiff())
optprob = GalacticOptim.instantiate_function(optf, x0, GalacticOptim.AutoFiniteDiff(), nothing)
optprob.grad(G2, x0)
@test G1 ≈ G2 rtol = 1e-6
optprob.hess(H2, x0)
@test H1 ≈ H2 rtol = 1e-6

prob = OptimizationProblem(optprob, x0)
sol = solve(prob, Optim.BFGS())
@test 10 * sol.minimum < l1

sol = solve(prob, Optim.Newton())
@test 10 * sol.minimum < l1

sol = solve(prob, Optim.KrylovTrustRegion())
@test sol.minimum < l1 #the loss doesn't go below 5e-1 here

sol = solve(prob, Optimisers.ADAM(0.1), maxiters=1000)
@test 10 * sol.minimum < l1
