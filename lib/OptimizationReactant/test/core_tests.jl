using ADTypes: AutoSparse
using CommonSolve: solve
using DifferentiationInterface: SecondOrder
using ForwardDiff: ForwardDiff
using OptimizationBase: OptimizationBase
using OptimizationOptimisers
using OptimizationReactant
using Optimisers
using Reactant: Reactant
using SciMLBase: OptimizationFunction, OptimizationProblem, OptimizationSolution, remake
using Test

@testset "AutoReactant instantiation" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    p = [1.0, 100.0]
    ref = ForwardDiff.gradient(x -> rosenbrock(x, p), x0)

    optf = OptimizationFunction(rosenbrock, AutoReactant())

    @testset "in-place OptimizationFunction{true}" begin
        f = OptimizationBase.instantiate_function(
            optf, x0, AutoReactant(), p, 0; g = true, fg = true
        )
        G = zeros(2)
        f.grad(G, x0, p)
        @test G ≈ ref
        @test f.fg(G, x0, p) ≈ rosenbrock(x0, p)
        @test G ≈ ref
        # `p` is a runtime argument, not baked into the compiled program.
        p2 = [2.0, 50.0]
        f.grad(G, x0, p2)
        @test G ≈ ForwardDiff.gradient(x -> rosenbrock(x, p2), x0)
        @test f.f(x0, p) ≈ rosenbrock(x0, p)
        @test f.f(x0) ≈ rosenbrock(x0, p)
        @test f.sys === optf.sys
    end

    @testset "out-of-place OptimizationFunction{false}" begin
        optf_oop = OptimizationFunction{false}(rosenbrock, AutoReactant())
        f = OptimizationBase.instantiate_function(
            optf_oop, x0, AutoReactant(), p, 0; g = true, fg = true
        )
        @test f.grad(x0, p) ≈ ref
        y, G = f.fg(x0, p)
        @test y ≈ rosenbrock(x0, p)
        @test G ≈ ref
    end

    @testset "ConcreteRArray arguments" begin
        f = OptimizationBase.instantiate_function(
            optf, x0, AutoReactant(), p, 0; g = true, fg = true
        )
        rx0 = Reactant.to_rarray(x0)
        rp = Reactant.to_rarray(p)
        G = copy(rx0)
        f.grad(G, rx0, rp)
        @test G isa Reactant.AbstractConcreteArray
        @test Array(G) ≈ ref
        @test f.f(rx0, rp) ≈ rosenbrock(x0, p)
        # Mutating the parameter buffers in place is observed by the compiled
        # gradient (the resample! contract).
        rxs = copy(rp)
        rxs .= [3.0, 25.0]
        f.grad(G, rx0, rxs)
        @test Array(G) ≈ ForwardDiff.gradient(x -> rosenbrock(x, [3.0, 25.0]), x0)
    end

    @testset "structured parameters" begin
        # MTKParameters-like `p`: array buffers plus host-side callables that
        # `to_rarray` must convert field-wise while leaving the callable alone.
        struct Params{F, A}
            apply::F
            tunable::A
            consts::A
        end
        obj_struct(x, p) = sum(abs2, x .- p.apply(p.tunable)) + sum(p.consts)
        p_struct = Params(v -> v .^ 2, [1.0, 2.0], [3.0, 4.0])
        ref_struct = ForwardDiff.gradient(x -> obj_struct(x, p_struct), x0)

        optf_s = OptimizationFunction(obj_struct, AutoReactant())
        f = OptimizationBase.instantiate_function(
            optf_s, x0, AutoReactant(), p_struct, 0; g = true
        )
        G = zeros(2)
        f.grad(G, x0, p_struct)
        @test G ≈ ref_struct

        # Buffer mutation under the same parameter object is observed.
        p_struct.tunable .= [5.0, -1.0]
        f.grad(G, x0, p_struct)
        @test G ≈ ForwardDiff.gradient(x -> obj_struct(x, p_struct), x0)

        rp_struct = Reactant.to_rarray(p_struct)
        @test rp_struct.apply === p_struct.apply
        Gr = copy(Reactant.to_rarray(x0))
        f.grad(Gr, Reactant.to_rarray(x0), rp_struct)
        @test Array(Gr) ≈ ForwardDiff.gradient(x -> obj_struct(x, p_struct), x0)
    end

    @testset "unsupported derivative requests error clearly" begin
        @test_throws ArgumentError OptimizationBase.instantiate_function(
            optf, x0, AutoReactant(), p, 0; h = true
        )
        @test_throws ArgumentError OptimizationBase.instantiate_function(
            optf, x0, AutoReactant(), p, 0; hv = true
        )
        optf_cons = OptimizationFunction(
            rosenbrock, AutoReactant();
            cons = (res, x, p) -> (res .= x)
        )
        @test_throws ArgumentError OptimizationBase.instantiate_function(
            optf_cons, x0, AutoReactant(), p, 1; cons_j = true
        )
        @test_throws ArgumentError OptimizationBase.instantiate_function(
            optf, x0, AutoSparse(AutoReactant()), p, 0
        )
        @test_throws ArgumentError OptimizationBase.instantiate_function(
            optf, x0, SecondOrder(AutoReactant(), AutoReactant()), p, 0
        )
    end

    @testset "end-to-end solve" begin
        quadratic(x, p) = sum(abs2, x .- p)
        optf_q = OptimizationFunction(quadratic, AutoReactant())
        prob = OptimizationProblem(optf_q, zeros(3), [1.0, 2.0, 3.0])
        sol = solve(prob, Optimisers.Adam(0.1); maxiters = 300)
        @test sol isa OptimizationSolution
        @test sol.objective < 1.0e-2

        prob_r = remake(prob; u0 = Reactant.to_rarray(prob.u0), p = Reactant.to_rarray(prob.p))
        sol_r = solve(prob_r, Optimisers.Adam(0.1); maxiters = 300)
        @test Array(sol_r.u) ≈ [1.0, 2.0, 3.0] atol = 1.0e-1
    end
end
