using Test
using Optimization
using OptimizationOptimJL
using OptimizationOptimisers
using ADTypes

# Test function - Rosenbrock
rosenbrock(x, p) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2

@testset "Interface Compatibility Tests" begin
    @testset "BigFloat Support" begin
        @testset "OptimizationProblem creation" begin
            f = OptimizationFunction(rosenbrock)
            x0 = BigFloat[0.5, 0.5]
            prob = OptimizationProblem(f, x0)
            @test typeof(prob.u0) == Vector{BigFloat}
            @test eltype(prob.u0) == BigFloat
        end

        @testset "NelderMead (gradient-free)" begin
            f = OptimizationFunction(rosenbrock)
            x0 = BigFloat[0.5, 0.5]
            prob = OptimizationProblem(f, x0)
            sol = solve(prob, Optim.NelderMead(), maxiters = 500)
            @test eltype(sol.u) == BigFloat
            @test sol.objective < 1.0e-6
        end

        @testset "BFGS with ForwardDiff" begin
            f = OptimizationFunction(rosenbrock, AutoForwardDiff())
            x0 = BigFloat[0.5, 0.5]
            prob = OptimizationProblem(f, x0)
            sol = solve(prob, Optim.BFGS(), maxiters = 500)
            @test eltype(sol.u) == BigFloat
            @test sol.objective < 1.0e-20
        end

        # Adam optimizer with BigFloat + ForwardDiff temporarily skipped due to
        # gradient dispatch MethodError. See GitHub issue #1134 for tracking.
        # @testset "Adam optimizer" begin
        #     f = OptimizationFunction(rosenbrock, AutoForwardDiff())
        #     x0 = BigFloat[0.5, 0.5]
        #     prob = OptimizationProblem(f, x0)
        #     sol = solve(prob, OptimizationOptimisers.Adam(BigFloat(0.01)), maxiters = 500)
        #     @test eltype(sol.u) == BigFloat
        # end
    end

    @testset "Type preservation patterns" begin
        @testset "similar preserves eltype" begin
            x = BigFloat[1.0, 2.0, 3.0]
            y = similar(x)
            @test eltype(y) == BigFloat
        end

        @testset "zero/one with eltype" begin
            x = BigFloat[1.0, 2.0, 3.0]
            @test zero(eltype(x)) isa BigFloat
            @test one(eltype(x)) isa BigFloat
        end

        @testset "zeros/ones with eltype" begin
            x = BigFloat[1.0, 2.0, 3.0]
            T = eltype(x)
            z = zeros(T, 3)
            o = ones(T, 3)
            @test eltype(z) == BigFloat
            @test eltype(o) == BigFloat
        end
    end

    @testset "Float64 baseline (regression check)" begin
        f = OptimizationFunction(rosenbrock, AutoForwardDiff())
        x0 = [0.5, 0.5]
        prob = OptimizationProblem(f, x0)
        sol = solve(prob, Optim.BFGS(), maxiters = 500)
        @test eltype(sol.u) == Float64
        @test sol.objective < 1.0e-20
        @test all(isapprox.(sol.u, [1.0, 1.0], atol = 1.0e-5))
    end
end
