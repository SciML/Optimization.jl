using Test
using Optimization
using OptimizationBase

# Test objective: f(x, p) = x[1]^3 + 5*x[2] + 6
# At x = [0.6, 0.9]:
#   f       = 0.216 + 4.5 + 6 = 10.716
#   ∇f      = [3*x[1]^2, 5] = [1.08, 5.0]
#   ∇²f     = [[6*x[1], 0], [0, 0]] = [[3.6, 0], [0, 0]]
#   (∇²f)v  = [[3.6, 0], [0, 0]] * [1.0, 2.0] = [3.6, 0.0]

const X0 = [0.6, 0.9]
const P0 = nothing

@testset "apply_sense" begin
    # --- Hand-written IIP derivative functions ---
    obj(x, p) = x[1]^3 + 5 * x[2] + 6

    function grad!(G, x, p)
        G[1] = 3 * x[1]^2
        G[2] = 5.0
    end

    function fg!(G, x, p)
        G[1] = 3 * x[1]^2
        G[2] = 5.0
        return x[1]^3 + 5 * x[2] + 6
    end

    function hess!(H, x, p)
        H[1, 1] = 6 * x[1]
        H[1, 2] = 0.0
        H[2, 1] = 0.0
        H[2, 2] = 0.0
    end

    function fgh!(G, H, x, p)
        G[1] = 3 * x[1]^2
        G[2] = 5.0
        H[1, 1] = 6 * x[1]
        H[1, 2] = 0.0
        H[2, 1] = 0.0
        H[2, 2] = 0.0
        return x[1]^3 + 5 * x[2] + 6
    end

    function hv!(Hv, x, v, p)
        # (∇²f) * v = [[6x[1], 0], [0, 0]] * v
        Hv[1] = 6 * x[1] * v[1]
        Hv[2] = 0.0
    end

    function lag_h!(H, x, σ, μ, p)
        # Lagrangian: σ*f(x) + μ[1]*c(x), where c(x) = x[1] + x[2]
        # ∇²L = σ * [[6x[1], 0], [0, 0]] + μ[1] * [[0,0],[0,0]]
        #      = σ * [[6x[1], 0], [0, 0]]
        H[1, 1] = σ * 6 * x[1]
        H[1, 2] = 0.0
        H[2, 1] = 0.0
        H[2, 2] = 0.0
    end

    optf = OptimizationFunction(obj;
        grad = grad!, fg = fg!, hess = hess!,
        fgh = fgh!, hv = hv!, lag_h = lag_h!
    )

    @testset "MinSense returns original function unchanged" begin
        f_min = OptimizationBase.apply_sense(optf, Optimization.MinSense)
        @test f_min === optf
    end

    @testset "MaxSense negates objective" begin
        f_max = OptimizationBase.apply_sense(optf, Optimization.MaxSense)
        @test f_max.f(X0, P0) ≈ -10.716
    end

    @testset "MaxSense negates gradient (IIP)" begin
        f_max = OptimizationBase.apply_sense(optf, Optimization.MaxSense)
        G = zeros(2)
        f_max.grad(G, X0, P0)
        @test G ≈ [-1.08, -5.0]
    end

    @testset "MaxSense negates fg (IIP)" begin
        f_max = OptimizationBase.apply_sense(optf, Optimization.MaxSense)
        G = zeros(2)
        y = f_max.fg(G, X0, P0)
        @test y ≈ -10.716
        @test G ≈ [-1.08, -5.0]
    end

    @testset "MaxSense negates Hessian (IIP)" begin
        f_max = OptimizationBase.apply_sense(optf, Optimization.MaxSense)
        H = zeros(2, 2)
        f_max.hess(H, X0, P0)
        @test H ≈ [[-3.6 0.0]; [0.0 0.0]]
    end

    @testset "MaxSense negates fgh (IIP)" begin
        f_max = OptimizationBase.apply_sense(optf, Optimization.MaxSense)
        G = zeros(2)
        H = zeros(2, 2)
        y = f_max.fgh(G, H, X0, P0)
        @test y ≈ -10.716
        @test G ≈ [-1.08, -5.0]
        @test H ≈ [[-3.6 0.0]; [0.0 0.0]]
    end

    @testset "MaxSense negates Hessian-vector product (IIP)" begin
        f_max = OptimizationBase.apply_sense(optf, Optimization.MaxSense)
        Hv = zeros(2)
        v = [1.0, 2.0]
        f_max.hv(Hv, X0, v, P0)
        @test Hv ≈ [-3.6, 0.0]
    end

    @testset "MaxSense negates σ in Lagrangian Hessian (IIP)" begin
        f_max = OptimizationBase.apply_sense(optf, Optimization.MaxSense)
        H = zeros(2, 2)
        σ = 1.0
        μ = [2.0]
        f_max.lag_h(H, X0, σ, μ, P0)
        # apply_sense passes -σ to the original lag_h
        # so H = (-1) * 6*0.6 = -3.6 in [1,1]
        @test H ≈ [[-3.6 0.0]; [0.0 0.0]]
    end

    @testset "nothing derivatives stay nothing under MaxSense" begin
        optf_bare = OptimizationFunction(obj)
        f_max = OptimizationBase.apply_sense(optf_bare, Optimization.MaxSense)
        @test f_max.grad === nothing
        @test f_max.fg === nothing
        @test f_max.hess === nothing
        @test f_max.fgh === nothing
        @test f_max.hv === nothing
        @test f_max.lag_h === nothing
    end

    @testset "Constraints are unchanged under MaxSense" begin
        cons!(res, x, p) = (res[1] = x[1] + x[2])
        optf_cons = OptimizationFunction(obj; cons = cons!)
        f_max = OptimizationBase.apply_sense(optf_cons, Optimization.MaxSense)
        res_orig = zeros(1)
        res_max = zeros(1)
        optf_cons.cons(res_orig, X0, P0)
        f_max.cons(res_max, X0, P0)
        @test res_orig == res_max
    end
end
