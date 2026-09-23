using ADTypes: AutoSparse
using CommonSolve: solve
using DifferentiationInterface: SecondOrder
using ForwardDiff: ForwardDiff
using OptimizationBase: OptimizationBase
using OptimizationOptimJL
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

    @testset "second-order objective derivatives" begin
        Href = ForwardDiff.hessian(x -> rosenbrock(x, p), x0)
        v = [0.3, -0.7]

        f = OptimizationBase.instantiate_function(
            optf, x0, AutoReactant(), p, 0; h = true, hv = true, fgh = true
        )
        H = zeros(2, 2)
        f.hess(H, x0, p)
        @test H ≈ Href
        res = zeros(2)
        f.hv(res, x0, v, p)
        @test res ≈ Href * v
        G = zeros(2)
        H2 = zeros(2, 2)
        y = f.fgh(G, H2, x0, p)
        @test y ≈ rosenbrock(x0, p)
        @test G ≈ ref
        @test H2 ≈ Href
        # `p` is a runtime argument for second derivatives too.
        p2 = [2.0, 50.0]
        f.hess(H, x0, p2)
        @test H ≈ ForwardDiff.hessian(x -> rosenbrock(x, p2), x0)

        optf_oop = OptimizationFunction{false}(rosenbrock, AutoReactant())
        f_oop = OptimizationBase.instantiate_function(
            optf_oop, x0, AutoReactant(), p, 0; h = true, hv = true, fgh = true
        )
        @test f_oop.hess(x0, p) ≈ Href
        @test f_oop.hv(x0, v, p) ≈ Href * v
        y2, G2, H3 = f_oop.fgh(x0, p)
        @test y2 ≈ rosenbrock(x0, p)
        @test G2 ≈ ref
        @test H3 ≈ Href
    end

    @testset "constraint derivatives" begin
        # `sin(x[1]) * x[2]` keeps a nonzero constraint Hessian while avoiding
        # a bare `x[i] * x[j]` product, which Reactant canonicalizes into a
        # `stablehlo.reduce` that Enzyme cannot differentiate.
        xc = [0.5, 0.3]
        cons_iip(res, x, p) = (
            res[1] = x[1]^2 + x[2]^2; res[2] = sin(x[1]) * x[2] - p[1]; nothing
        )
        cons_oop(x, p) = vcat(x[1]^2 + x[2]^2, sin(x[1]) * x[2] - p[1])
        Jref = ForwardDiff.jacobian(x -> cons_oop(x, p), xc)
        Hc = [ForwardDiff.hessian(x -> cons_oop(x, p)[i], xc) for i in 1:2]
        σ, λ, v, w = 1.7, [0.4, -0.9], [0.3, -0.7], [0.6, 0.2]

        optf_c = OptimizationFunction(rosenbrock, AutoReactant(); cons = cons_iip)
        f = OptimizationBase.instantiate_function(
            optf_c, x0, AutoReactant(), p, 2;
            cons_j = true, cons_vjp = true, cons_jvp = true,
            cons_h = true, lag_h = true
        )
        J = zeros(2, 2)
        f.cons_j(J, xc, p)
        @test J ≈ Jref
        vjp = zeros(2)
        f.cons_vjp(vjp, xc, w)
        @test vjp ≈ Jref' * w
        jvp = zeros(2)
        f.cons_jvp(jvp, xc, v)
        @test jvp ≈ Jref * v
        Hs = [zeros(2, 2), zeros(2, 2)]
        f.cons_h(Hs, xc)
        @test Hs[1] ≈ Hc[1]
        @test Hs[2] ≈ Hc[2]
        Lref = σ * ForwardDiff.hessian(x -> rosenbrock(x, p), xc) +
            λ[1] * Hc[1] + λ[2] * Hc[2]
        L = zeros(2, 2)
        f.lag_h(L, xc, σ, λ, p)
        @test L ≈ Lref
        Lv = zeros(3)
        f.lag_h(Lv, xc, σ, λ, p)
        @test Lv ≈ [Lref[1, 1], Lref[2, 1], Lref[2, 2]]

        optf_co = OptimizationFunction{false}(
            rosenbrock, AutoReactant(); cons = cons_oop
        )
        f_oop = OptimizationBase.instantiate_function(
            optf_co, x0, AutoReactant(), p, 2;
            cons_j = true, cons_vjp = true, cons_jvp = true,
            cons_h = true, lag_h = true
        )
        @test f_oop.cons_j(xc, p) ≈ Jref
        @test f_oop.cons_vjp(xc, w) ≈ Jref' * w
        @test f_oop.cons_jvp(xc, v) ≈ Jref * v
        Hso = f_oop.cons_h(xc)
        @test Hso[1] ≈ Hc[1]
        @test Hso[2] ≈ Hc[2]
        @test f_oop.lag_h(xc, σ, λ, p) ≈ Lref
    end

    @testset "user-supplied derivatives are preserved" begin
        mygrad(res, x, p) = (res .= 42 .* ones(2))
        optf_g = OptimizationFunction(rosenbrock, AutoReactant(); grad = mygrad)
        f = OptimizationBase.instantiate_function(
            optf_g, x0, AutoReactant(), p, 0; g = true, h = true
        )
        G = zeros(2)
        f.grad(G, x0, p)
        @test G == fill(42.0, 2)
        H = zeros(2, 2)
        f.hess(H, x0, p)
        @test H ≈ ForwardDiff.hessian(x -> rosenbrock(x, p), x0)
    end

    @testset "unsupported derivative schemes error clearly" begin
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
        # On Julia 1.10, Reactant's broadcast `copy` infers a non-concrete eltype for
        # Adam's nested update after `Broadcast.flatten` and throws.
        @test Array(solve(prob_r, Optimisers.Adam(0.1); maxiters = 300).u) ≈
            [1.0, 2.0, 3.0] atol = 1.0e-1 broken = VERSION < v"1.11"
    end

    @testset "second-order solver paths" begin
        quadratic(x, p) = sum(abs2, x .- p)
        optf_q = OptimizationFunction(quadratic, AutoReactant())
        prob = OptimizationProblem(optf_q, [1.0, 2.0], [3.0, 4.0])
        sol = solve(prob, OptimizationOptimJL.Optim.Newton())
        @test sol.u ≈ [3.0, 4.0] atol = 1.0e-6

        cons(res, x, p) = (res[1] = x[1]^2 + x[2]^2)
        optf_c = OptimizationFunction(quadratic, AutoReactant(); cons = cons)
        prob_c = OptimizationProblem(
            optf_c, [1.0, 2.0], [3.0, 4.0]; lcons = [-Inf], ucons = [4.0]
        )
        sol_c = solve(prob_c, OptimizationOptimJL.Optim.IPNewton())
        res = zeros(1)
        cons(res, sol_c.u, nothing)
        @test res[1] ≈ 4.0 rtol = 1.0e-3
        @test sol_c.objective ≈ 9.0 rtol = 1.0e-3
    end
end
