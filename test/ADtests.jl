using Optimization, OptimizationOptimJL, OptimizationOptimisers, Test
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
optf = OptimizationFunction(rosenbrock, Optimization.AutoModelingToolkit(), cons=cons)
optprob = Optimization.instantiate_function(optf, x0, Optimization.AutoModelingToolkit(), nothing, 1)
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
optf = OptimizationFunction(rosenbrock, Optimization.AutoModelingToolkit(), cons=con2_c)
optprob = Optimization.instantiate_function(optf, x0, Optimization.AutoModelingToolkit(), nothing, 2)
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

optf = OptimizationFunction(rosenbrock, Optimization.AutoModelingToolkit(true, true), cons=con2_c)
optprob = Optimization.instantiate_function(optf, x0, Optimization.AutoModelingToolkit(true, true), nothing, 2)
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

optf = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
optprob = Optimization.instantiate_function(optf, x0, Optimization.AutoForwardDiff(), nothing)
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

optf = OptimizationFunction(rosenbrock, Optimization.AutoZygote())
optprob = Optimization.instantiate_function(optf, x0, Optimization.AutoZygote(), nothing)
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

optf = OptimizationFunction(rosenbrock, Optimization.AutoReverseDiff())
optprob = Optimization.instantiate_function(optf, x0, Optimization.AutoReverseDiff(), nothing)
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

optf = OptimizationFunction(rosenbrock, Optimization.AutoTracker())
optprob = Optimization.instantiate_function(optf, x0, Optimization.AutoTracker(), nothing)
optprob.grad(G2, x0)
@test G1 == G2
@test_throws ErrorException optprob.hess(H2, x0)


prob = OptimizationProblem(optprob, x0)

sol = solve(prob, Optim.BFGS())
@test 10 * sol.minimum < l1

@test_throws ErrorException solve(prob, Newton())

optf = OptimizationFunction(rosenbrock, Optimization.AutoFiniteDiff())
optprob = Optimization.instantiate_function(optf, x0, Optimization.AutoFiniteDiff(), nothing)
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

# Test new constraints
cons = (x, p) -> [x[1]^2 + x[2]^2]
optf = OptimizationFunction(rosenbrock, Optimization.AutoFiniteDiff(), cons=cons)
optprob = Optimization.instantiate_function(optf, x0, Optimization.AutoFiniteDiff(), nothing, 1)
optprob.grad(G2, x0)
@test G1 ≈ G2 rtol = 1e-6
optprob.hess(H2, x0)
@test H1 ≈ H2 rtol = 1e-6
@test optprob.cons(x0) == [0.0]
@test optprob.cons([1.0, 4.0]) == [17.0]
J = zeros(1, 2)
optprob.cons_j(J, [5.0, 3.0])
@test J ≈ [10.0 6.0]
H3 = [Array{Float64}(undef, 2, 2)]
optprob.cons_h(H3, x0)
@test H3 ≈ [[2.0 0.0; 0.0 2.0]]

cons_jac_proto = Float64.(sparse([1 1])) # Things break if you only use [1 1]; see FiniteDiff.jl
cons_jac_colors = 1:2 
optf = OptimizationFunction(rosenbrock, Optimization.AutoFiniteDiff(), cons=cons, cons_jac_prototype = cons_jac_proto, cons_jac_colorvec = cons_jac_colors)
optprob = Optimization.instantiate_function(optf, x0, Optimization.AutoFiniteDiff(), nothing, 1)
@test optprob.cons_jac_prototype == sparse([1.0 1.0]) # make sure it's still using it 
@test optprob.cons_jac_colorvec == 1:2
J = zeros(1, 2)
optprob.cons_j(J, [5.0, 3.0])
@test J ≈ [10.0 6.0]

function con2_c(x, p)
    [x[1]^2 + x[2]^2, x[2] * sin(x[1]) - x[1]]
end
optf = OptimizationFunction(rosenbrock, Optimization.AutoFiniteDiff(), cons=con2_c)
optprob = Optimization.instantiate_function(optf, x0, Optimization.AutoFiniteDiff(), nothing, 2)
optprob.grad(G2, x0)
@test G1 ≈ G2 rtol = 1e-6
optprob.hess(H2, x0)
@test H1 ≈ H2 rtol = 1e-6
@test optprob.cons(x0) == [0.0, 0.0]
@test optprob.cons([1.0, 2.0]) ≈ [5.0, 0.682941969615793]
J = Array{Float64}(undef, 2, 2)
optprob.cons_j(J, [5.0, 3.0])
@test all(isapprox(J, [10.0 6.0; -0.149013 -0.958924]; rtol=1e-3))
H3 = [Array{Float64}(undef, 2, 2), Array{Float64}(undef, 2, 2)]
optprob.cons_h(H3, x0)
@test H3 ≈ [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

cons_jac_proto = Float64.(sparse([1 1; 1 1])) 
cons_jac_colors = 1:2 
optf = OptimizationFunction(rosenbrock, Optimization.AutoFiniteDiff(), cons=con2_c, cons_jac_prototype = cons_jac_proto, cons_jac_colorvec = cons_jac_colors)
optprob = Optimization.instantiate_function(optf, x0, Optimization.AutoFiniteDiff(), nothing, 2)
@test optprob.cons_jac_prototype == sparse([1.0 1.0; 1.0 1.0]) # make sure it's still using it 
@test optprob.cons_jac_colorvec == 1:2
J = Array{Float64}(undef, 2, 2)
optprob.cons_j(J, [5.0, 3.0])
@test all(isapprox(J, [10.0 6.0; -0.149013 -0.958924]; rtol=1e-3))

# Can we solve problems? Using AutoForwardDiff to test since we know that works 
for consf in [cons, con2_c]
    optf1 = OptimizationFunction(rosenbrock, Optimization.AutoFiniteDiff(); cons = consf)
    prob1 = OptimizationProblem(optf1, [0.3, 0.5], lb = [0.2, 0.4], ub = [0.6, 0.8])
    sol1 = solve(prob1,BFGS())
    optf2 = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff(); cons = consf)
    prob2 = OptimizationProblem(optf2, [0.3, 0.5], lb = [0.2, 0.4], ub = [0.6, 0.8])
    sol2 = solve(prob2,BFGS())
    @test sol1.minimum ≈ sol2.minimum 
    @test sol1.u ≈ sol2.u

    optf1 = OptimizationFunction(rosenbrock, Optimization.AutoFiniteDiff(); cons = consf)
    lcons = consf == cons ? [0.2] : [0.2, 0.33]
    ucons = consf == cons ? [0.55] : [0.55, 0.81]
    prob1 = OptimizationProblem(optf1, [0.3, 0.5], lb = [0.2, 0.4], ub = [0.6, 0.8], lcons = lcons, ucons = ucons)
    sol1 = solve(prob1,IPNewton())
    optf2 = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff(); cons = consf)
    prob2 = OptimizationProblem(optf2, [0.3, 0.5], lb = [0.2, 0.4], ub = [0.6, 0.8], lcons = lcons, ucons = ucons)
    sol2 = solve(prob2,IPNewton())
    @test sol1.minimum ≈ sol2.minimum 
    @test sol1.u ≈ sol2.u
    @test lcons[1] ≤ consf(sol1.u, nothing)[1] ≤ ucons[1]
    if consf == con2_c
        @test lcons[2] ≤ consf(sol1.u, nothing)[2] ≤ ucons[2]
    end

    lcons = consf == cons ? [0.2] : [0.2, 0.4]
    ucons = consf == cons ? [0.2] : [0.2, 0.4]
    optf1 = OptimizationFunction(rosenbrock, Optimization.AutoFiniteDiff(); cons = consf)
    prob1 = OptimizationProblem(optf1, [0.3, 0.5], lcons = lcons, ucons = ucons)
    sol1 = solve(prob1,IPNewton(), maxiters=500)
    optf2 = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff(); cons = consf)
    prob2 = OptimizationProblem(optf2, [0.3, 0.5], lcons = lcons, ucons = ucons)
    sol2 = solve(prob2,IPNewton(), maxiters=500)
    @test sol1.minimum ≈ sol2.minimum 
    @test sol1.u ≈ sol2.u    
    @test consf(sol1.u, nothing)[1] ≈ lcons[1]
    if consf == con2_c
        @test consf(sol1.u, nothing)[2] ≈ lcons[2]
    end
end