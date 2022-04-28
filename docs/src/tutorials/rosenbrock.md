# Rosenbrock function examples

!!! note

    This example uses many different solvers of GalacticOptim.jl. Each solver
    subpackage needs to be installed separate. For example, for the details on 
    the installation and usage of GalacticOptimJL.jl package, see the 
    [Optim.jl page](@ref optim).

```julia
using GalacticOptim, Optim, ForwardDiff, Zygote, Test, Random

rosenbrock(x, p) =  (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
_p  = [1.0, 100.0]

f = OptimizationFunction(rosenbrock, GalacticOptim.AutoForwardDiff())
l1 = rosenbrock(x0, _p)
prob = OptimizationProblem(f, x0, _p)

## Optim.jl Solvers

using GalacticOptimJL

sol = solve(prob, SimulatedAnnealing())
@test 10*sol.minimum < l1

Random.seed!(1234)
prob = OptimizationProblem(f, x0, _p, lb=[-1.0, -1.0], ub=[0.8, 0.8])
sol = solve(prob, SAMIN())
@test 10*sol.minimum < l1

l1 = rosenbrock(x0)
prob = OptimizationProblem(rosenbrock, x0)
sol = solve(prob, NelderMead())
@test 10*sol.minimum < l1

cons= (x,p) -> [x[1]^2 + x[2]^2]
optprob = OptimizationFunction(rosenbrock, GalacticOptim.AutoForwardDiff();cons= cons)

prob = OptimizationProblem(optprob, x0)

sol = solve(prob, ADAM(0.1), maxiters = 1000)
@test 10*sol.minimum < l1

sol = solve(prob, BFGS())
@test 10*sol.minimum < l1

sol = solve(prob, Newton())
@test 10*sol.minimum < l1

sol = solve(prob, Optim.KrylovTrustRegion())
@test 10*sol.minimum < l1

prob = OptimizationProblem(optprob, x0, lcons = [-Inf], ucons = [Inf])
sol = solve(prob, IPNewton())
@test 10*sol.minimum < l1

prob = OptimizationProblem(optprob, x0, lcons = [-5.0], ucons = [10.0])
sol = solve(prob, IPNewton())
@test 10*sol.minimum < l1

prob = OptimizationProblem(optprob, x0, lcons = [-Inf], ucons = [Inf], lb = [-500.0,-500.0], ub=[50.0,50.0])
sol = solve(prob, IPNewton())
@test sol.minimum < l1

function con2_c(x,p)
    [x[1]^2 + x[2]^2, x[2]*sin(x[1])-x[1]]
end

optprob = OptimizationFunction(rosenbrock, GalacticOptim.AutoForwardDiff();cons= con2_c)
prob = OptimizationProblem(optprob, x0, lcons = [-Inf,-Inf], ucons = [Inf,Inf])
sol = solve(prob, IPNewton())
@test 10*sol.minimum < l1

cons_circ = (x,p) -> [x[1]^2 + x[2]^2]
optprob = OptimizationFunction(rosenbrock, GalacticOptim.AutoForwardDiff();cons= cons_circ)
prob = OptimizationProblem(optprob, x0, lcons = [-Inf], ucons = [0.25^2])
sol = solve(prob, IPNewton())
@test sqrt(cons(sol.minimizer,nothing)[1]) ≈ 0.25 rtol = 1e-6

optprob = OptimizationFunction(rosenbrock, GalacticOptim.AutoZygote())
prob = OptimizationProblem(optprob, x0)
sol = solve(prob, ADAM(), maxiters = 1000, progress = false)
@test 10*sol.minimum < l1

prob = OptimizationProblem(optprob, x0, lb=[-1.0, -1.0], ub=[0.8, 0.8])
sol = solve(prob, Fminbox())
@test 10*sol.minimum < l1

prob = OptimizationProblem(optprob, x0, lb=[-1.0, -1.0], ub=[0.8, 0.8])
@test_broken @test_nowarn sol = solve(prob, SAMIN())
@test 10*sol.minimum < l1

## CMAEvolutionStrategy.jl solvers

using GalacticCMAEvolutionStrategy
sol = solve(prob, CMAEvolutionStrategyOpt())
@test 10*sol.minimum < l1

rosenbrock(x, p=nothing) =  (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2

## NLopt.jl solvers

using GalacticNLopt
prob = OptimizationProblem(optprob, x0)
sol = solve(prob, Opt(:LN_BOBYQA, 2))
@test 10*sol.minimum < l1

sol = solve(prob, Opt(:LD_LBFGS, 2))
@test 10*sol.minimum < l1

prob = OptimizationProblem(optprob, x0, lb=[-1.0, -1.0], ub=[0.8, 0.8])
sol = solve(prob, Opt(:LD_LBFGS, 2))
@test 10*sol.minimum < l1

sol = solve(prob, Opt(:G_MLSL_LDS, 2), nstart=2, local_method = Opt(:LD_LBFGS, 2), maxiters=10000)
@test 10*sol.minimum < l1

## Evolutionary.jl Solvers

using GalacticEvolutionary
sol = solve(prob, CMAES(μ =40 , λ = 100),abstol=1e-15)
@test 10*sol.minimum < l1

## BlackBoxOptim.jl Solvers

using GalacticBBO
prob = GalacticOptim.OptimizationProblem(optprob, x0, lb=[-1.0, -1.0], ub=[0.8, 0.8])
sol = solve(prob, BBO())
@test 10*sol.minimum < l1
```
