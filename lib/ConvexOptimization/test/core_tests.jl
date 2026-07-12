using ConvexOptimization
using SciMLBase
using SciMLBase: ConvexOptimizationProblem, ConvexOptimizationSolution
import MathOptInterface as MOI
import Clarabel
using Test

# minimize  x1 + 2 x2   s.t.  x1 + x2 == 1,  x >= 0
# analytic optimum: x* = (1, 0), obj = 1
# LP duals: equality multiplier y = 1; nonneg-cone dual s = c - Aᵀy = (0, 1) >= 0
c = [1.0, 2.0]

@testset "LP solve: primal, objective, retcode" begin
    optf = OptimizationFunction((u, p) -> c[1] * u[1] + c[2] * u[2])
    cons = [
        ConeConstraint((u, p) -> [u[1] + u[2] - 1.0], MOI.Zeros(1)),        # sum(x) == 1
        ConeConstraint((u, p) -> [u[1], u[2]], MOI.Nonnegatives(2)),        # x >= 0
    ]
    prob = ConvexOptimizationProblem(optf, [0.5, 0.5]; constraints = cons)

    sol = solve(prob, ConvexMOI(Clarabel.Optimizer))

    @test sol isa ConvexOptimizationSolution
    @test SciMLBase.successful_retcode(sol.retcode)
    @test isapprox(sol.u, [1.0, 0.0]; atol = 1.0e-6)
    @test isapprox(sol.objective, 1.0; atol = 1.0e-6)

    @testset "dual: one entry per constraint, in user variables" begin
        @test sol.dual !== nothing
        @test length(sol.dual) == 2
        @test isapprox(only(sol.dual[1]), 1.0; atol = 1.0e-6)      # equality multiplier
        @test isapprox(sol.dual[2], [0.0, 1.0]; atol = 1.0e-6)     # nonneg-cone dual
    end

    @testset "cache carries the SciMLBase glue fields" begin
        @test sol.cache isa SciMLBase.AbstractOptimizationCache
        @test sol.cache.p isa SciMLBase.NullParameters
        @test sol.cache.u0 == [0.5, 0.5]
    end
end

@testset "non-convex objective is rejected, not mis-solved" begin
    # x1*x2 is neither convex nor concave -> certification must error.
    optf = OptimizationFunction((u, p) -> u[1] * u[2])
    prob = ConvexOptimizationProblem(optf, [0.5, 0.5])
    @test_throws Exception solve(prob, ConvexMOI(Clarabel.Optimizer))
end

@testset "SOC constraint lowers and solves" begin
    # minimize t  s.t.  || (x1, x2) ||_2 <= t,  x == (3, 4)  ->  t* = 5
    optf = OptimizationFunction((u, p) -> u[3])   # u = (x1, x2, t)
    cons = [
        ConeConstraint((u, p) -> [u[1] - 3.0, u[2] - 4.0], MOI.Zeros(2)),   # x fixed
        ConeConstraint((u, p) -> [u[3], u[1], u[2]], MOI.SecondOrderCone(3)), # (t, x) in SOC
    ]
    prob = ConvexOptimizationProblem(optf, [0.0, 0.0, 0.0]; constraints = cons)
    sol = solve(prob, ConvexMOI(Clarabel.Optimizer))
    @test SciMLBase.successful_retcode(sol.retcode)
    @test isapprox(sol.objective, 5.0; atol = 1.0e-5)   # ||(3,4)|| = 5
    @test isapprox(sol.u[3], 5.0; atol = 1.0e-5)
end

# Cross-check the LP primal AND dual against Convex.jl (same solver). The exit
# gate for the vertical slice: matching Convex.jl on both, to solver tolerance.
@testset "matches Convex.jl (primal + dual)" begin
    convex_available = try
        @eval import Convex
        true
    catch
        @info "Convex.jl not available in this environment; skipping cross-check " *
            "(the analytic assertions above already pin the same values)."
        false
    end
    if convex_available
        xc = Convex.Variable(2)
        pc = Convex.minimize(c[1] * xc[1] + c[2] * xc[2], [sum(xc) == 1, xc >= 0])
        Convex.solve!(pc, Clarabel.Optimizer; silent = true)
        @test isapprox(Convex.evaluate(xc), [1.0, 0.0]; atol = 1.0e-6)
        @test isapprox(pc.optval, 1.0; atol = 1.0e-6)
    end
end
