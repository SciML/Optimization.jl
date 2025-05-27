
using Test
using Optimization
using Optimization.SciMLBase
using Optimization.ADTypes
using OptimizationODE

function quad(u, p)
    return sum(u .^ 2) + p[1] * u[2]^2
end

function quad_grad!(g, u, p, data)
    g[1] = 2u[1]
    g[2] = 2p[1]*u[2]
    return g
end

function rosenbrock(u, p)
    return (p[1] - u[1])^2 + p[2]*(u[2] - u[1]^2)^2
end

function rosenbrock_grad!(g, u, p, data)
    g[1] = -2(p[1] - u[1]) - 4p[2]*u[1]*(u[2] - u[1]^2)
    g[2] = 2p[2]*(u[2] - u[1]^2)
    return g
end

@testset "OptimizationODE Solvers" begin
    u0q = [2.0, -3.0]
    pq = [5.0]
    fq = OptimizationFunction(quad, SciMLBase.NoAD(); grad = quad_grad!)
    probQ = OptimizationProblem(fq, u0q, pq)

    @testset "ODEGradientDescent on Quadratic" begin
        sol = solve(probQ, ODEGradientDescent; dt = 0.1, maxiters = 1000)
        @test isapprox(sol.u, zeros(length(sol.u)); atol=1e-2)
    end

    @testset "RKChebyshevDescent on Quadratic" begin
        sol = solve(probQ, RKChebyshevDescent; dt = 0.1, maxiters = 1000)
        @test isapprox(sol.u, zeros(length(sol.u)); atol=1e-2)
    end

    @testset "RKAccelerated on Quadratic" begin
        sol = solve(probQ, RKAccelerated; dt = 0.1, maxiters = 1000)
        @test isapprox(sol.u, zeros(length(sol.u)); atol=1e-2)
    end

    @testset "PRKChebyshevDescent on Quadratic" begin
        sol = solve(probQ, PRKChebyshevDescent; dt = 0.1, maxiters = 1000)
        @test isapprox(sol.u, zeros(length(sol.u)); atol=1e-2)
    end

    u0r = [-1.2, 1.0]
    pr  = [1.0, 100.0]

    for (mode, desc) in [(SciMLBase.NoAD(), "NoAD"),
                         (ADTypes.AutoForwardDiff(), "AutoForwardDiff")]
        @testset "ODEGradientDescent on Rosenbrock ($desc)" begin
            fr = OptimizationFunction(rosenbrock, mode; grad = rosenbrock_grad!)
            probR = OptimizationProblem(fr, u0r, pr)
            sol = solve(probR, ODEGradientDescent; dt = 0.001, maxiters = 5000)
            @test isapprox(sol.u[1], 1.0; atol=1e-1)
            @test isapprox(sol.u[2], 1.0; atol=2e-1)
        end
    end
end
