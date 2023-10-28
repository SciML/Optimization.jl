using OptimizationSolvers, ForwardDiff, Optimization
using Test
using Zygote


@testset "OptimizationSolvers.jl" begin
    function objf(x, p)
        return x[1]^2 + x[2]^2
    end

    optprob = OptimizationFunction(objf, Optimization.AutoZygote())
    x0 = zeros(2) .+ 1
    prob = OptimizationProblem(optprob, x0)
    l1 = objf(x0, nothing)
    sol = Optimization.solve(prob,
        OptimizationSolvers.BFGS(1e-3, 10),
        maxiters = 10)
    @test 10 * sol.objective < l1

    sol = Optimization.solve(prob,
        OptimizationSolvers.LBFGS(1e-3, 10),
        maxiters = 50)

    x0 = zeros(2)
    rosenbrock(x, p = nothing) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
    l1 = rosenbrock(x0)
    optf = OptimizationFunction(rosenbrock, Optimization.AutoZygote())
    prob = OptimizationProblem(optf, [-1.2, 1.0])
    sol = Optimization.solve(prob,
        OptimizationSolvers.BFGS(1e-5),
        maxiters = 100)
    @test 10 * sol.objective < l1

    sol = Optimization.solve(prob,
        OptimizationSolvers.LBFGS(1e-3, 10),
        maxiters = 50)


end
