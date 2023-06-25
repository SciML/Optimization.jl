
using SparseDiffTools

optf = OptimizationFunction(rosenbrock, Optimization.AutoFiniteDiff(), cons = con2_c)
optprob = Optimization.instantiate_function(optf, x0, Optimization.AutoFiniteDiff(),
    nothing, 2)
G2 = Array{Float64}(undef, 2)
optprob.grad(G2, x0)
@test G1≈G2 rtol=1e-4
H2 = Array{Float64}(undef, 2, 2)
optprob.hess(H2, x0)
@test H1≈H2 rtol=1e-4
res = Array{Float64}(undef, 2)
optprob.cons(res, x0)
@test res≈[0.0, 0.0] atol=1e-4
optprob.cons(res, [1.0, 2.0])
@test res ≈ [5.0, 0.682941969615793]
J = Array{Float64}(undef, 2, 2)
optprob.cons_j(J, [5.0, 3.0])
@test all(isapprox(J, [10.0 6.0; -0.149013 -0.958924]; rtol = 1e-3))
H3 = [Array{Float64}(undef, 2, 2), Array{Float64}(undef, 2, 2)]
optprob.cons_h(H3, x0)
@test H3 ≈ [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

optf = OptimizationFunction(rosenbrock, Optimization.AutoFiniteDiff())
optprob = Optimization.instantiate_function(optf, x0, Optimization.AutoFiniteDiff(),
    nothing)
optprob.grad(G2, x0)
@test G1≈G2 rtol=1e-6
optprob.hess(H2, x0)
@test H1≈H2 rtol=1e-4

prob = OptimizationProblem(optf, x0)
sol = solve(prob, Optim.BFGS())
@test 10 * sol.objective < l1

sol = solve(prob, Optim.Newton())
@test 10 * sol.objective < l1

#at 0,0 it gives error because of the inaccuracy of the hessian and hv calculations
prob = OptimizationProblem(optf, x0 + rand(2))
sol = solve(prob, Optim.KrylovTrustRegion())
@test sol.objective < l1

sol = solve(prob, Optimisers.ADAM(0.1), maxiters = 1000)
@test 10 * sol.objective < l1

optf = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff(), cons = con2_c)
optprob = Optimization.instantiate_function(optf, x0, Optimization.AutoForwardDiff(),
    nothing, 2)
G2 = Array{Float64}(undef, 2)
optprob.grad(G2, x0)
@test G1≈G2 rtol=1e-4
H2 = Array{Float64}(undef, 2, 2)
optprob.hess(H2, x0)
@test H1≈H2 rtol=1e-4
res = Array{Float64}(undef, 2)
optprob.cons(res, x0)
@test res≈[0.0, 0.0] atol=1e-4
optprob.cons(res, [1.0, 2.0])
@test res ≈ [5.0, 0.682941969615793]
J = Array{Float64}(undef, 2, 2)
optprob.cons_j(J, [5.0, 3.0])
@test all(isapprox(J, [10.0 6.0; -0.149013 -0.958924]; rtol = 1e-3))
H3 = [Array{Float64}(undef, 2, 2), Array{Float64}(undef, 2, 2)]
optprob.cons_h(H3, x0)
@test H3 ≈ [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

optf = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
optprob = Optimization.instantiate_function(optf, x0, Optimization.AutoForwardDiff(),
    nothing)
optprob.grad(G2, x0)
@test G1≈G2 rtol=1e-6
optprob.hess(H2, x0)
@test H1≈H2 rtol=1e-6

prob = OptimizationProblem(optf, x0)
sol = solve(prob, Optim.BFGS())
@test 10 * sol.objective < l1

sol = solve(prob, Optim.Newton())
@test 10 * sol.objective < l1

sol = solve(prob, Optim.KrylovTrustRegion())
@test sol.objective < l1

sol = solve(prob, Optimisers.ADAM(0.1), maxiters = 1000)
@test 10 * sol.objective < l1
