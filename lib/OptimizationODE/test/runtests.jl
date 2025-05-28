using Test
using Optimization
using Optimization.SciMLBase
using Optimization.ADTypes
using OptimizationODE

quad(u, p) = u[1]^2 + p[1]*u[2]^2
function quad_grad!(g, u, p, data)
    g[1] = 2u[1]
    g[2] = 2p[1]*u[2]
    return g
end

rosen(u, p) = (p[1] - u[1])^2 + p[2]*(u[2] - u[1]^2)^2
function rosen_grad!(g, u, p, data)
    g[1] = -2*(p[1] - u[1]) - 4*p[2]*u[1]*(u[2] - u[1]^2)
    g[2] =  2*p[2]*(u[2] - u[1]^2)
    return g
end

make_zeros(u) = zero.(u)
make_ones(u)  = fill(one(eltype(u)), length(u))

@testset "OptimizationODE: Steady‐State Solvers" begin

    u0q = [2.0, -3.0]
    pq  = [5.0]
    fq  = OptimizationFunction(quad, SciMLBase.NoAD(); grad = quad_grad!)
    probQ_noad = OptimizationProblem(fq, u0q, pq)

    u0r = [-1.2, 1.0]
    pr  = [1.0, 100.0]

    ADmodes = (
        (SciMLBase.NoAD(),       "NoAD",       nothing),
        (ADTypes.AutoForwardDiff(), "ForwardDiff", nothing)
    )


    @testset "ODEGradientDescent on Quadratic" begin
        sol = solve(probQ_noad, ODEGradientDescent; dt = 0.1, maxiters = 2000)
        @test isapprox(sol.u, make_zeros(sol.u); atol = 1e-2)
        @test sol.retcode == ReturnCode.Success
    end

    @testset "ODEGradientDescent on Rosenbrock" begin
        for (ad, name, _) in ADmodes
            fr = OptimizationFunction(rosen, ad; grad = rosen_grad!)
            probR = OptimizationProblem(fr, u0r, pr)
            sol = solve(probR, ODEGradientDescent; dt = 0.001, maxiters = 3000)
            @test isapprox(sol.u, make_ones(sol.u); atol = 0.01)
            @test sol.retcode == ReturnCode.Success
        end
    end


    @testset "RKChebyshevDescent on Quadratic" begin
        sol = solve(probQ_noad, RKChebyshevDescent; dt = 0.1, maxiters = 1000)
        @test isapprox(sol.u, make_zeros(sol.u); atol = 1e-2)
        @test sol.retcode == ReturnCode.Success
    end

    @testset "RKAccelerated on Quadratic" begin
        sol = solve(probQ_noad, RKAccelerated; dt = 0.1, maxiters = 1000)
        @test isapprox(sol.u, make_zeros(sol.u); atol = 1e-2)
        @test sol.retcode == ReturnCode.Success
    end


    @testset "PRKChebyshevDescent on Quadratic" begin
        sol = solve(probQ_noad, PRKChebyshevDescent; dt = 0.1, maxiters = 1000)
        @test isapprox(sol.u, make_zeros(sol.u); atol = 1e-2)
        @test sol.retcode == ReturnCode.Success
    end

    @testset "PRKChebyshevDescent on Rosenbrock (NoAD)" begin
        fr = OptimizationFunction(rosen, SciMLBase.NoAD(); grad = rosen_grad!)
        probR = OptimizationProblem(fr, u0r, pr)
        sol = solve(probR, PRKChebyshevDescent; dt = 0.001, maxiters = 2000)
        @test isapprox(sol.u, make_ones(sol.u); atol = 1e-1)
        @test sol.retcode == ReturnCode.Success
    end

end
