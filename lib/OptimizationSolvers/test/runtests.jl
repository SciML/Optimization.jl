using OptimizationSolvers, ForwardDiff, Optimization
using Test
using Zygote

@testset "OptimizationOptimisers.jl" begin
    
    function objf(x, p)
        return x[1]^2 + x[2]^2 + 2*x[1]* x[2]
    end

    optprob = OptimizationFunction(objf, Optimization.AutoZygote())
    x0 = zeros(2) .+ 1
    x0[1] = 0.5 
    prob = OptimizationProblem(optprob, x0)
    
    sol = Optimization.solve(prob,
        OptimizationSolvers.BFGS(1e-3, 5),
        maxiters = 1000)
    @test 10 * sol.objective < l1

    prob = OptimizationProblem(optprob, x0)
    
end
