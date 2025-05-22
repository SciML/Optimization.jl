using Test
using Optimization
using Optimization.SciMLBase
using OptimizationODE

@testset "ODEGradientDescent Tests" begin

    # Define the Rosenbrock objective and its gradient
    function rosen(u, p)
        return (p[1] - u[1])^2 + p[2] * (u[2] - u[1]^2)^2
    end

    function rosen_grad!(g, u, p, data)
        g[1] = -2 * (p[1] - u[1]) - 4 * p[2] * u[1] * (u[2] - u[1]^2)
        g[2] =  2 * p[2] * (u[2] - u[1]^2)
        return g
    end

    # Set up the problem
    u0 = [0.0, 0.0]
    p  = [1.0, 100.0]

    # Wrap into an OptimizationFunction without AD, providing our gradient
    f = OptimizationFunction(
        rosen,
        Optimization.SciMLBase.NoAD();
        grad = rosen_grad!
    )

    prob = OptimizationProblem(f, u0, p)

    # Solve with ODEGradientDescent 
    sol = solve(
        prob,
        ODEGradientDescent();
        η    = 0.001,
        tmax = 1_000.0,
        dt   = 0.01
    )

    # Assertions 
    @test isapprox(sol.u[1], 1.0; atol = 1e-2)
    @test isapprox(sol.u[2], 1.0; atol = 1e-2)

end
