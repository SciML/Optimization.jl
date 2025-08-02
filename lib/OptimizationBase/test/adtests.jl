using OptimizationBase, Test, DifferentiationInterface, SparseArrays, Symbolics
using ForwardDiff, Zygote, ReverseDiff, FiniteDiff, Tracker
using ModelingToolkit, Enzyme, Random

x0 = zeros(2)
rosenbrock(x, p = nothing) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
l1 = rosenbrock(x0)

function g!(G, x)
    G[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    G[2] = 200.0 * (x[2] - x[1]^2)
end

function h!(H, x)
    H[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
    H[1, 2] = -400.0 * x[1]
    H[2, 1] = -400.0 * x[1]
    H[2, 2] = 200.0
end

G1 = Array{Float64}(undef, 2)
G2 = Array{Float64}(undef, 2)
H1 = Array{Float64}(undef, 2, 2)
H2 = Array{Float64}(undef, 2, 2)

g!(G1, x0)
h!(H1, x0)

cons = (res, x, p) -> (res[1] = x[1]^2 + x[2]^2; return nothing)
optf = OptimizationFunction(rosenbrock, OptimizationBase.AutoModelingToolkit(), cons = cons)
optprob = OptimizationBase.instantiate_function(optf, x0,
    OptimizationBase.AutoModelingToolkit(),
    nothing, 1, g = true, h = true, cons_j = true, cons_h = true)
optprob.grad(G2, x0)
@test G1 == G2
optprob.hess(H2, x0)
@test H1 == H2
res = Array{Float64}(undef, 1)
optprob.cons(res, x0)
@test res == [0.0]
J = Array{Float64}(undef, 2)
optprob.cons_j(J, [5.0, 3.0])
@test J == [10.0, 6.0]
H3 = [Array{Float64}(undef, 2, 2)]
optprob.cons_h(H3, x0)
@test H3 == [[2.0 0.0; 0.0 2.0]]

function con2_c(res, x, p)
    res[1] = x[1]^2 + x[2]^2
    res[2] = x[2] * sin(x[1]) - x[1]
    return nothing
end
optf = OptimizationFunction(rosenbrock,
    OptimizationBase.AutoModelingToolkit(),
    cons = con2_c)
optprob = OptimizationBase.instantiate_function(optf, x0,
    OptimizationBase.AutoModelingToolkit(),
    nothing, 2, g = true, h = true, cons_j = true, cons_h = true)
optprob.grad(G2, x0)
@test G1 == G2
optprob.hess(H2, x0)
@test H1 == H2
res = Array{Float64}(undef, 2)
optprob.cons(res, x0)
@test res == [0.0, 0.0]
J = Array{Float64}(undef, 2, 2)
optprob.cons_j(J, [5.0, 3.0])
@test all(isapprox(J, [10.0 6.0; -0.149013 -0.958924]; rtol = 1e-3))
H3 = [Array{Float64}(undef, 2, 2), Array{Float64}(undef, 2, 2)]
optprob.cons_h(H3, x0)
@test H3 == [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

@testset "one constraint tests" begin
    G2 = Array{Float64}(undef, 2)
    H2 = Array{Float64}(undef, 2, 2)
    optf = OptimizationFunction(rosenbrock, OptimizationBase.AutoEnzyme(), cons = cons)
    optprob = OptimizationBase.instantiate_function(
        optf, x0, OptimizationBase.AutoEnzyme(),
        nothing, 1, g = true, h = true, hv = true,
        cons_j = true, cons_h = true, cons_vjp = true,
        cons_jvp = true, lag_h = true)
    optprob.grad(G2, x0)
    @test G1 == G2
    optprob.hess(H2, x0)
    @test H1 == H2
    Hv = Array{Float64}(undef, 2)
    optprob.hv(Hv, x0, [1.0, 1.0])
    @test Hv == [2.0, 200.0]
    res = Array{Float64}(undef, 1)
    optprob.cons(res, x0)
    @test res == [0.0]
    J = Array{Float64}(undef, 2)
    optprob.cons_j(J, [5.0, 3.0])
    @test J == [10.0, 6.0]
    vJ = Array{Float64}(undef, 2)
    optprob.cons_vjp(vJ, [5.0, 3.0], [1.0])
    @test vJ == [10.0, 6.0]
    Jv = Array{Float64}(undef, 1)
    optprob.cons_jvp(Jv, [5.0, 3.0], [0.5, 0.5])
    @test Jv == [8.0]
    H3 = [Array{Float64}(undef, 2, 2)]
    optprob.cons_h(H3, x0)
    @test H3 == [[2.0 0.0; 0.0 2.0]]
    H4 = Array{Float64}(undef, 2, 2)
    μ = randn(1)
    σ = rand()
    optprob.lag_h(H4, x0, σ, μ)
    @test H4≈σ * H2 + μ[1] * H3[1] rtol=1e-6

    G2 = Array{Float64}(undef, 2)
    H2 = Array{Float64}(undef, 2, 2)

    optf = OptimizationFunction(rosenbrock, OptimizationBase.AutoForwardDiff(), cons = cons)
    optprob = OptimizationBase.instantiate_function(
        optf, x0, OptimizationBase.AutoForwardDiff(),
        nothing, 1, g = true, h = true, hv = true,
        cons_j = true, cons_h = true, cons_vjp = true,
        cons_jvp = true, lag_h = true)
    optprob.grad(G2, x0)
    @test G1 == G2
    optprob.hess(H2, x0)
    @test H1 == H2
    Hv = Array{Float64}(undef, 2)
    optprob.hv(Hv, x0, [1.0, 1.0])
    @test Hv == [2.0, 200.0]
    res = Array{Float64}(undef, 1)
    optprob.cons(res, x0)
    @test res == [0.0]
    J = Array{Float64}(undef, 2)
    optprob.cons_j(J, [5.0, 3.0])
    @test J == [10.0, 6.0]
    vJ = Array{Float64}(undef, 2)
    optprob.cons_vjp(vJ, [5.0, 3.0], [1.0])
    @test vJ == [10.0, 6.0]
    Jv = Array{Float64}(undef, 1)
    optprob.cons_jvp(Jv, [5.0, 3.0], [0.5, 0.5])
    @test Jv == [8.0]
    H3 = [Array{Float64}(undef, 2, 2)]
    optprob.cons_h(H3, x0)
    @test H3 == [[2.0 0.0; 0.0 2.0]]
    H4 = Array{Float64}(undef, 2, 2)
    μ = randn(1)
    σ = rand()
    optprob.lag_h(H4, x0, σ, μ)
    @test H4≈σ * H2 + μ[1] * H3[1] rtol=1e-6

    G2 = Array{Float64}(undef, 2)
    H2 = Array{Float64}(undef, 2, 2)

    optf = OptimizationFunction(rosenbrock, OptimizationBase.AutoReverseDiff(), cons = cons)
    optprob = OptimizationBase.instantiate_function(
        optf, x0, OptimizationBase.AutoReverseDiff(),
        nothing, 1, g = true, h = true, hv = true,
        cons_j = true, cons_h = true, cons_vjp = true,
        cons_jvp = true, lag_h = true)
    optprob.grad(G2, x0)
    @test G1 == G2
    optprob.hess(H2, x0)
    @test H1 == H2
    Hv = Array{Float64}(undef, 2)
    optprob.hv(Hv, x0, [1.0, 1.0])
    @test Hv == [2.0, 200.0]
    res = Array{Float64}(undef, 1)
    optprob.cons(res, x0)
    @test res == [0.0]
    J = Array{Float64}(undef, 2)
    optprob.cons_j(J, [5.0, 3.0])
    @test J == [10.0, 6.0]
    vJ = Array{Float64}(undef, 2)
    optprob.cons_vjp(vJ, [5.0, 3.0], [1.0])
    @test vJ == [10.0, 6.0]
    Jv = Array{Float64}(undef, 1)
    optprob.cons_jvp(Jv, [5.0, 3.0], [0.5, 0.5])
    @test Jv == [8.0]
    H3 = [Array{Float64}(undef, 2, 2)]
    optprob.cons_h(H3, x0)
    @test H3 == [[2.0 0.0; 0.0 2.0]]
    H4 = Array{Float64}(undef, 2, 2)
    μ = randn(1)
    σ = rand()
    optprob.lag_h(H4, x0, σ, μ)
    @test H4≈σ * H2 + μ[1] * H3[1] rtol=1e-6

    G2 = Array{Float64}(undef, 2)
    H2 = Array{Float64}(undef, 2, 2)

    optf = OptimizationFunction(
        rosenbrock, OptimizationBase.AutoReverseDiff(true), cons = cons)
    optprob = OptimizationBase.instantiate_function(
        optf, x0, OptimizationBase.AutoReverseDiff(true),
        nothing, 1, g = true, h = true, hv = true,
        cons_j = true, cons_h = true, cons_vjp = true,
        cons_jvp = true, lag_h = true)
    optprob.grad(G2, x0)
    @test G1 == G2
    optprob.hess(H2, x0)
    @test H1 == H2
    Hv = Array{Float64}(undef, 2)
    optprob.hv(Hv, x0, [1.0, 1.0])
    @test Hv == [2.0, 200.0]
    res = Array{Float64}(undef, 1)
    optprob.cons(res, x0)
    @test res == [0.0]
    J = Array{Float64}(undef, 2)
    optprob.cons_j(J, [5.0, 3.0])
    @test J == [10.0, 6.0]
    vJ = Array{Float64}(undef, 2)
    optprob.cons_vjp(vJ, [5.0, 3.0], [1.0])
    @test vJ == [10.0, 6.0]
    Jv = Array{Float64}(undef, 1)
    optprob.cons_jvp(Jv, [5.0, 3.0], [0.5, 0.5])
    @test Jv == [8.0]
    H3 = [Array{Float64}(undef, 2, 2)]
    optprob.cons_h(H3, x0)
    @test H3 == [[2.0 0.0; 0.0 2.0]]
    H4 = Array{Float64}(undef, 2, 2)
    μ = randn(1)
    σ = rand()
    optprob.lag_h(H4, x0, σ, μ)
    @test H4≈σ * H2 + μ[1] * H3[1] rtol=1e-6

    G2 = Array{Float64}(undef, 2)
    H2 = Array{Float64}(undef, 2, 2)

    optf = OptimizationFunction(
        rosenbrock, AutoZygote(), cons = cons)
    optprob = OptimizationBase.instantiate_function(
        optf, x0, AutoZygote(),
        nothing, 1, g = true, h = true, hv = true,
        cons_j = true, cons_h = true, cons_vjp = true,
        cons_jvp = true, lag_h = true)
    optprob.grad(G2, x0)
    @test G1 == G2
    optprob.hess(H2, x0)
    @test H1 == H2
    Hv = Array{Float64}(undef, 2)
    optprob.hv(Hv, x0, [1.0, 1.0])
    @test Hv == [2.0, 200.0]
    res = Array{Float64}(undef, 1)
    optprob.cons(res, x0)
    @test res == [0.0]
    J = Array{Float64}(undef, 2)
    optprob.cons_j(J, [5.0, 3.0])
    @test J == [10.0, 6.0]
    vJ = Array{Float64}(undef, 2)
    optprob.cons_vjp(vJ, [5.0, 3.0], [1.0])
    @test vJ == [10.0, 6.0]
    Jv = Array{Float64}(undef, 1)
    optprob.cons_jvp(Jv, [5.0, 3.0], [0.5, 0.5])
    @test Jv == [8.0]
    H3 = [Array{Float64}(undef, 2, 2)]
    optprob.cons_h(H3, x0)
    @test H3 == [[2.0 0.0; 0.0 2.0]]
    H4 = Array{Float64}(undef, 2, 2)
    μ = randn(1)
    σ = rand()
    optprob.lag_h(H4, x0, σ, μ)
    @test H4≈σ * H2 + μ[1] * H3[1] rtol=1e-6

    G2 = Array{Float64}(undef, 2)
    H2 = Array{Float64}(undef, 2, 2)

    optf = OptimizationFunction(rosenbrock,
        DifferentiationInterface.SecondOrder(
            ADTypes.AutoFiniteDiff(), ADTypes.AutoReverseDiff()),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(
        optf, x0,
        DifferentiationInterface.SecondOrder(
            ADTypes.AutoFiniteDiff(), ADTypes.AutoReverseDiff()),
        nothing, 1, g = true, h = true, hv = true,
        cons_j = true, cons_h = true, cons_vjp = true,
        cons_jvp = true, lag_h = true)
    optprob.grad(G2, x0)
    @test G1≈G2 rtol=1e-5
    optprob.hess(H2, x0)
    @test H1≈H2 rtol=1e-5
    Hv = Array{Float64}(undef, 2)
    optprob.hv(Hv, x0, [1.0, 1.0])
    @test Hv≈[2.0, 200.0] rtol=1e-5
    res = Array{Float64}(undef, 1)
    optprob.cons(res, x0)
    @test res ≈ [0.0]
    J = Array{Float64}(undef, 1, 2)
    optprob.cons_j(J, [5.0, 3.0])
    @test J≈[10.0 6.0] rtol=1e-5
    vJ = Array{Float64}(undef, 2)
    optprob.cons_vjp(vJ, [5.0, 3.0], [1.0])
    @test vJ≈[10.0, 6.0] rtol=1e-5
    Jv = Array{Float64}(undef, 1)
    optprob.cons_jvp(Jv, [5.0, 3.0], [0.5, 0.5])
    @test Jv≈[8.0] rtol=1e-5
    H3 = [Array{Float64}(undef, 2, 2)]
    optprob.cons_h(H3, x0)
    @test H3≈[[2.0 0.0; 0.0 2.0]] rtol=1e-5
    Random.seed!(123)
    H4 = Array{Float64}(undef, 2, 2)
    μ = randn(1)
    σ = rand()
    optprob.lag_h(H4, x0, σ, μ)
    @test H4≈σ * H2 + μ[1] * H3[1] rtol=1e-6
end

@testset "two constraints tests" begin
    G2 = Array{Float64}(undef, 2)
    H2 = Array{Float64}(undef, 2, 2)
    optf = OptimizationFunction(rosenbrock, OptimizationBase.AutoEnzyme(), cons = con2_c)
    optprob = OptimizationBase.instantiate_function(
        optf, x0, OptimizationBase.AutoEnzyme(),
        nothing, 2, g = true, h = true, hv = true,
        cons_j = true, cons_h = true, cons_vjp = true,
        cons_jvp = true, lag_h = true)
    optprob.grad(G2, x0)
    @test G1 == G2
    optprob.hess(H2, x0)
    @test H1 == H2
    Hv = Array{Float64}(undef, 2)
    optprob.hv(Hv, x0, [1.0, 1.0])
    @test Hv == [2.0, 200.0]
    res = Array{Float64}(undef, 2)
    optprob.cons(res, x0)
    @test res == [0.0, 0.0]
    J = Array{Float64}(undef, 2, 2)
    optprob.cons_j(J, [5.0, 3.0])
    @test all(isapprox(J, [10.0 6.0; -0.149013 -0.958924]; rtol = 1e-3))
    vJ = Array{Float64}(undef, 2)
    optprob.cons_vjp(vJ, [5.0, 3.0], [1.0, 1.0])
    @test vJ == sum(J, dims = 1)[:]
    Jv = Array{Float64}(undef, 2)
    optprob.cons_jvp(Jv, [5.0, 3.0], [0.5, 0.5])
    @test Jv ≈ 0.5 * sum(J, dims = 2)[:]
    H3 = [Array{Float64}(undef, 2, 2), Array{Float64}(undef, 2, 2)]
    optprob.cons_h(H3, x0)
    @test H3 == [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]
    H4 = Array{Float64}(undef, 2, 2)
    μ = randn(2)
    σ = rand()
    optprob.lag_h(H4, x0, σ, μ)
    @test H4≈σ * H1 + sum(μ .* H3) rtol=1e-6

    G2 = Array{Float64}(undef, 2)
    H2 = Array{Float64}(undef, 2, 2)

    optf = OptimizationFunction(
        rosenbrock, OptimizationBase.AutoReverseDiff(), cons = con2_c)
    optprob = OptimizationBase.instantiate_function(optf, x0,
        OptimizationBase.AutoReverseDiff(),
        nothing, 2, g = true, h = true, hv = true,
        cons_j = true, cons_h = true, cons_vjp = true,
        cons_jvp = true, lag_h = true)
    optprob.grad(G2, x0)
    @test G1 == G2
    optprob.hess(H2, x0)
    @test H1 == H2
    Hv = Array{Float64}(undef, 2)
    optprob.hv(Hv, x0, [1.0, 1.0])
    @test Hv == [2.0, 200.0]
    res = Array{Float64}(undef, 2)
    optprob.cons(res, x0)
    @test res == [0.0, 0.0]
    J = Array{Float64}(undef, 2, 2)
    optprob.cons_j(J, [5.0, 3.0])
    @test all(isapprox(J, [10.0 6.0; -0.149013 -0.958924]; rtol = 1e-3))
    vJ = Array{Float64}(undef, 2)
    optprob.cons_vjp(vJ, [5.0, 3.0], [1.0, 1.0])
    @test vJ == sum(J, dims = 1)[:]
    Jv = Array{Float64}(undef, 2)
    optprob.cons_jvp(Jv, [5.0, 3.0], [0.5, 0.5])
    @test Jv == 0.5 * sum(J, dims = 2)[:]
    H3 = [Array{Float64}(undef, 2, 2), Array{Float64}(undef, 2, 2)]
    optprob.cons_h(H3, x0)
    @test H3 == [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]
    H4 = Array{Float64}(undef, 2, 2)
    μ = randn(2)
    σ = rand()
    optprob.lag_h(H4, x0, σ, μ)
    @test H4≈σ * H1 + sum(μ .* H3) rtol=1e-6

    G2 = Array{Float64}(undef, 2)
    H2 = Array{Float64}(undef, 2, 2)

    optf = OptimizationFunction(
        rosenbrock, OptimizationBase.AutoReverseDiff(true), cons = con2_c)
    optprob = OptimizationBase.instantiate_function(optf, x0,
        OptimizationBase.AutoReverseDiff(true),
        nothing, 2, g = true, h = true, hv = true,
        cons_j = true, cons_h = true, cons_vjp = true,
        cons_jvp = true, lag_h = true)
    optprob.grad(G2, x0)
    @test G1 == G2
    optprob.hess(H2, x0)
    @test H1 == H2
    Hv = Array{Float64}(undef, 2)
    optprob.hv(Hv, x0, [1.0, 1.0])
    @test Hv == [2.0, 200.0]
    res = Array{Float64}(undef, 2)
    optprob.cons(res, x0)
    @test res == [0.0, 0.0]
    J = Array{Float64}(undef, 2, 2)
    optprob.cons_j(J, [5.0, 3.0])
    @test all(isapprox(J, [10.0 6.0; -0.149013 -0.958924]; rtol = 1e-3))
    vJ = Array{Float64}(undef, 2)
    optprob.cons_vjp(vJ, [5.0, 3.0], [1.0, 1.0])
    @test vJ == sum(J, dims = 1)[:]
    Jv = Array{Float64}(undef, 2)
    optprob.cons_jvp(Jv, [5.0, 3.0], [0.5, 0.5])
    @test Jv == 0.5 * sum(J, dims = 2)[:]
    H3 = [Array{Float64}(undef, 2, 2), Array{Float64}(undef, 2, 2)]
    optprob.cons_h(H3, x0)
    @test H3 == [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]
    H4 = Array{Float64}(undef, 2, 2)
    μ = randn(2)
    σ = rand()
    optprob.lag_h(H4, x0, σ, μ)
    @test H4≈σ * H1 + sum(μ .* H3) rtol=1e-6

    G2 = Array{Float64}(undef, 2)
    H2 = Array{Float64}(undef, 2, 2)

    optf = OptimizationFunction(
        rosenbrock, OptimizationBase.AutoForwardDiff(), cons = con2_c)
    optprob = OptimizationBase.instantiate_function(optf, x0,
        OptimizationBase.AutoReverseDiff(compile = true),
        nothing, 2, g = true, h = true, hv = true,
        cons_j = true, cons_h = true, cons_vjp = true,
        cons_jvp = true, lag_h = true)
    optprob.grad(G2, x0)
    @test G1 == G2
    optprob.hess(H2, x0)
    @test H1 == H2
    Hv = Array{Float64}(undef, 2)
    optprob.hv(Hv, x0, [1.0, 1.0])
    @test Hv == [2.0, 200.0]
    res = Array{Float64}(undef, 2)
    optprob.cons(res, x0)
    @test res == [0.0, 0.0]
    J = Array{Float64}(undef, 2, 2)
    optprob.cons_j(J, [5.0, 3.0])
    @test all(isapprox(J, [10.0 6.0; -0.149013 -0.958924]; rtol = 1e-3))
    vJ = Array{Float64}(undef, 2)
    optprob.cons_vjp(vJ, [5.0, 3.0], [1.0, 1.0])
    @test vJ == sum(J, dims = 1)[:]
    Jv = Array{Float64}(undef, 2)
    optprob.cons_jvp(Jv, [5.0, 3.0], [0.5, 0.5])
    @test Jv == 0.5 * sum(J, dims = 2)[:]
    H3 = [Array{Float64}(undef, 2, 2), Array{Float64}(undef, 2, 2)]
    optprob.cons_h(H3, x0)
    @test H3 == [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]
    H4 = Array{Float64}(undef, 2, 2)
    μ = randn(2)
    σ = rand()
    optprob.lag_h(H4, x0, σ, μ)
    @test H4≈σ * H1 + sum(μ .* H3) rtol=1e-6

    G2 = Array{Float64}(undef, 2)
    H2 = Array{Float64}(undef, 2, 2)

    optf = OptimizationFunction(
        rosenbrock, AutoZygote(), cons = con2_c)
    optprob = OptimizationBase.instantiate_function(
        optf, x0, AutoZygote(),
        nothing, 2, g = true, h = true, hv = true,
        cons_j = true, cons_h = true, cons_vjp = true,
        cons_jvp = true, lag_h = true)
    optprob.grad(G2, x0)
    @test G1 == G2
    optprob.hess(H2, x0)
    @test H1 == H2
    Hv = Array{Float64}(undef, 2)
    optprob.hv(Hv, x0, [1.0, 1.0])
    @test Hv == [2.0, 200.0]
    res = Array{Float64}(undef, 2)
    optprob.cons(res, x0)
    @test res == [0.0, 0.0]
    J = Array{Float64}(undef, 2, 2)
    optprob.cons_j(J, [5.0, 3.0])
    @test all(isapprox(J, [10.0 6.0; -0.149013 -0.958924]; rtol = 1e-3))
    vJ = Array{Float64}(undef, 2)
    optprob.cons_vjp(vJ, [5.0, 3.0], [1.0, 1.0])
    @test vJ == sum(J, dims = 1)[:]
    Jv = Array{Float64}(undef, 2)
    optprob.cons_jvp(Jv, [5.0, 3.0], [0.5, 0.5])
    @test Jv == 0.5 * sum(J, dims = 2)[:]
    H3 = [Array{Float64}(undef, 2, 2), Array{Float64}(undef, 2, 2)]
    optprob.cons_h(H3, x0)
    @test H3 == [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]
    H4 = Array{Float64}(undef, 2, 2)
    μ = randn(2)
    σ = rand()
    optprob.lag_h(H4, x0, σ, μ)
    @test H4≈σ * H1 + sum(μ .* H3) rtol=1e-6

    G2 = Array{Float64}(undef, 2)
    H2 = Array{Float64}(undef, 2, 2)

    optf = OptimizationFunction(
        rosenbrock, DifferentiationInterface.SecondOrder(
            ADTypes.AutoFiniteDiff(), ADTypes.AutoReverseDiff()),
        cons = con2_c)
    optprob = OptimizationBase.instantiate_function(
        optf, x0,
        DifferentiationInterface.SecondOrder(
            ADTypes.AutoFiniteDiff(), ADTypes.AutoReverseDiff()),
        nothing, 2, g = true, h = true, hv = true,
        cons_j = true, cons_h = true, cons_vjp = true,
        cons_jvp = true, lag_h = true)
    optprob.grad(G2, x0)
    @test G1≈G2 rtol=1e-5
    optprob.hess(H2, x0)
    @test H1≈H2 rtol=1e-5
    Hv = Array{Float64}(undef, 2)
    optprob.hv(Hv, x0, [1.0, 1.0])
    @test Hv≈[2.0, 200.0] rtol=1e-5
    res = Array{Float64}(undef, 2)
    optprob.cons(res, x0)
    @test res ≈ [0.0, 0.0]
    J = Array{Float64}(undef, 2, 2)
    optprob.cons_j(J, [5.0, 3.0])
    @test all(isapprox(J, [10.0 6.0; -0.149013 -0.958924]; rtol = 1e-3))
    vJ = Array{Float64}(undef, 2)
    optprob.cons_vjp(vJ, [5.0, 3.0], [1.0, 1.0])
    @test vJ≈sum(J, dims = 1)[:] rtol=1e-5
    Jv = Array{Float64}(undef, 2)
    optprob.cons_jvp(Jv, [5.0, 3.0], [0.5, 0.5])
    @test Jv≈0.5 * sum(J, dims = 2)[:] rtol=1e-5
    H3 = [Array{Float64}(undef, 2, 2), Array{Float64}(undef, 2, 2)]
    optprob.cons_h(H3, x0)
    @test H3≈[[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]] rtol=1e-5
    H4 = Array{Float64}(undef, 2, 2)
    μ = randn(2)
    σ = rand()
    optprob.lag_h(H4, x0, σ, μ)
    @test H4≈σ * H1 + sum(μ .* H3) rtol=1e-6
end

@testset "Sparse Tests" begin
    # Define a sparse objective function
    function sparse_objective(x, p)
        return x[1]^2 + 100 * (x[3] - x[2]^2)^2
    end

    # Define sparse constraints
    function sparse_constraints(res, x, p)
        res[1] = x[1] + x[2] + (x[2] * x[3])^2 - 1
        res[2] = x[1]^2 + x[3]^2 - 1
    end

    # Initial point
    x0 = [0.5, 0.5, 0.5]

    # Create OptimizationFunction
    optf = OptimizationFunction(sparse_objective, OptimizationBase.AutoSparseForwardDiff(),
        cons = sparse_constraints)

    # Instantiate the optimization problem
    optprob = OptimizationBase.instantiate_function(optf, x0,
        OptimizationBase.AutoSparseForwardDiff(),
        nothing, 2, g = true, h = true, cons_j = true, cons_h = true, lag_h = true)
    # Test gradient
    G = zeros(3)
    optprob.grad(G, x0)
    @test G ≈ [1.0, -50.0, 50.0]

    # Test Hessian
    H_expected = sparse(
        [1, 2, 2, 3, 3], [1, 2, 3, 2, 3], [2.0, 100.0, -200.0, -200.0, 200.0], 3, 3)
    H = similar(optprob.hess_prototype, Float64)
    optprob.hess(H, x0)
    @test H ≈ H_expected
    @test nnz(H) == 5  # Check sparsity

    # Test constraints
    res = zeros(2)
    optprob.cons(res, x0)
    @test res ≈ [0.0625, -0.5]

    # Test constraint Jacobian
    J_expected = sparse([1, 1, 1, 2, 2], [1, 2, 3, 1, 3], [1.0, 1.25, 0.25, 1.0, 1.0], 2, 3)
    J = similar(optprob.cons_jac_prototype, Float64)
    optprob.cons_j(J, x0)
    @test J ≈ J_expected
    @test nnz(J) == 5  # Check sparsity

    # Test constraint Hessians
    H_cons_expected = [sparse([2, 2, 3, 3], [2, 3, 2, 3], [0.5, 1.0, 1.0, 0.5], 3, 3),
        sparse([1, 3], [1, 3], [2.0, 2.0], 3, 3)]
    H_cons = [similar(h, Float64) for h in optprob.cons_hess_prototype]
    optprob.cons_h(H_cons, x0)
    @test all(H_cons .≈ H_cons_expected)
    @test all(nnz.(H_cons) .== [4, 2])  # Check sparsity

    lag_H_expected = sparse(
        [1, 2, 3, 2, 3], [1, 2, 2, 3, 3], [6.0, 100.5, -199.0, -199.0, 204.5], 3, 3)
    σ = 1.0
    λ = [1.0, 2.0]
    lag_H = similar(optprob.lag_hess_prototype, Float64)
    optprob.lag_h(lag_H, x0, σ, λ)
    @test lag_H ≈ lag_H_expected
    @test nnz(lag_H) == 5

    optf = OptimizationFunction(sparse_objective, OptimizationBase.AutoSparseReverseDiff(),
        cons = sparse_constraints)

    # Instantiate the optimization problem
    optprob = OptimizationBase.instantiate_function(optf, x0,
        OptimizationBase.AutoSparseForwardDiff(),
        nothing, 2, g = true, h = true, cons_j = true, cons_h = true, lag_h = true)
    # Test gradient
    G = zeros(3)
    optprob.grad(G, x0)
    @test G ≈ [1.0, -50.0, 50.0]

    # Test Hessian
    H_expected = sparse(
        [1, 2, 2, 3, 3], [1, 2, 3, 2, 3], [2.0, 100.0, -200.0, -200.0, 200.0], 3, 3)
    H = similar(optprob.hess_prototype, Float64)
    optprob.hess(H, x0)
    @test H ≈ H_expected
    @test nnz(H) == 5  # Check sparsity

    # Test constraints
    res = zeros(2)
    optprob.cons(res, x0)
    @test res ≈ [0.0625, -0.5]

    # Test constraint Jacobian
    J_expected = sparse([1, 1, 1, 2, 2], [1, 2, 3, 1, 3], [1.0, 1.25, 0.25, 1.0, 1.0], 2, 3)
    J = similar(optprob.cons_jac_prototype, Float64)
    optprob.cons_j(J, x0)
    @test J ≈ J_expected
    @test nnz(J) == 5  # Check sparsity

    # Test constraint Hessians
    H_cons_expected = [sparse([2, 2, 3, 3], [2, 3, 2, 3], [0.5, 1.0, 1.0, 0.5], 3, 3),
        sparse([1, 3], [1, 3], [2.0, 2.0], 3, 3)]
    H_cons = [similar(h, Float64) for h in optprob.cons_hess_prototype]
    optprob.cons_h(H_cons, x0)
    @test all(H_cons .≈ H_cons_expected)
    @test all(nnz.(H_cons) .== [4, 2])  # Check sparsity

    lag_H_expected = sparse(
        [1, 2, 3, 2, 3], [1, 2, 2, 3, 3], [6.0, 100.5, -199.0, -199.0, 204.5], 3, 3)
    σ = 1.0
    λ = [1.0, 2.0]
    lag_H = similar(optprob.lag_hess_prototype, Float64)
    optprob.lag_h(lag_H, x0, σ, λ)
    @test lag_H ≈ lag_H_expected
    @test nnz(lag_H) == 5

    optf = OptimizationFunction(
        sparse_objective, OptimizationBase.AutoSparseReverseDiff(true),
        cons = sparse_constraints)

    # Instantiate the optimization problem
    optprob = OptimizationBase.instantiate_function(optf, x0,
        OptimizationBase.AutoSparseForwardDiff(),
        nothing, 2, g = true, h = true, cons_j = true, cons_h = true, lag_h = true)
    # Test gradient
    G = zeros(3)
    optprob.grad(G, x0)
    @test G ≈ [1.0, -50.0, 50.0]

    # Test Hessian
    H_expected = sparse(
        [1, 2, 2, 3, 3], [1, 2, 3, 2, 3], [2.0, 100.0, -200.0, -200.0, 200.0], 3, 3)
    H = similar(optprob.hess_prototype, Float64)
    optprob.hess(H, x0)
    @test H ≈ H_expected
    @test nnz(H) == 5  # Check sparsity

    # Test constraints
    res = zeros(2)
    optprob.cons(res, x0)
    @test res ≈ [0.0625, -0.5]

    # Test constraint Jacobian
    J_expected = sparse([1, 1, 1, 2, 2], [1, 2, 3, 1, 3], [1.0, 1.25, 0.25, 1.0, 1.0], 2, 3)
    J = similar(optprob.cons_jac_prototype, Float64)
    optprob.cons_j(J, x0)
    @test J ≈ J_expected
    @test nnz(J) == 5  # Check sparsity

    # Test constraint Hessians
    H_cons_expected = [sparse([2, 2, 3, 3], [2, 3, 2, 3], [0.5, 1.0, 1.0, 0.5], 3, 3),
        sparse([1, 3], [1, 3], [2.0, 2.0], 3, 3)]
    H_cons = [similar(h, Float64) for h in optprob.cons_hess_prototype]
    optprob.cons_h(H_cons, x0)
    @test all(H_cons .≈ H_cons_expected)
    @test all(nnz.(H_cons) .== [4, 2])  # Check sparsity

    lag_H_expected = sparse(
        [1, 2, 3, 2, 3], [1, 2, 2, 3, 3], [6.0, 100.5, -199.0, -199.0, 204.5], 3, 3)
    σ = 1.0
    λ = [1.0, 2.0]
    lag_H = similar(optprob.lag_hess_prototype, Float64)
    optprob.lag_h(lag_H, x0, σ, λ)
    @test lag_H ≈ lag_H_expected
    @test nnz(lag_H) == 5

    optf = OptimizationFunction(sparse_objective, OptimizationBase.AutoSparseFiniteDiff(),
        cons = sparse_constraints)

    # Instantiate the optimization problem
    optprob = OptimizationBase.instantiate_function(optf, x0,
        OptimizationBase.AutoSparseForwardDiff(),
        nothing, 2, g = true, h = true, cons_j = true, cons_h = true, lag_h = true)
    # Test gradient
    G = zeros(3)
    optprob.grad(G, x0)
    @test G ≈ [1.0, -50.0, 50.0]

    # Test Hessian
    H_expected = sparse(
        [1, 2, 2, 3, 3], [1, 2, 3, 2, 3], [2.0, 100.0, -200.0, -200.0, 200.0], 3, 3)
    H = similar(optprob.hess_prototype, Float64)
    optprob.hess(H, x0)
    @test H ≈ H_expected
    @test nnz(H) == 5  # Check sparsity

    # Test constraints
    res = zeros(2)
    optprob.cons(res, x0)
    @test res ≈ [0.0625, -0.5]

    # Test constraint Jacobian
    J_expected = sparse([1, 1, 1, 2, 2], [1, 2, 3, 1, 3], [1.0, 1.25, 0.25, 1.0, 1.0], 2, 3)
    J = similar(optprob.cons_jac_prototype, Float64)
    optprob.cons_j(J, x0)
    @test J ≈ J_expected
    @test nnz(J) == 5  # Check sparsity

    # Test constraint Hessians
    H_cons_expected = [sparse([2, 2, 3, 3], [2, 3, 2, 3], [0.5, 1.0, 1.0, 0.5], 3, 3),
        sparse([1, 3], [1, 3], [2.0, 2.0], 3, 3)]
    H_cons = [similar(h, Float64) for h in optprob.cons_hess_prototype]
    optprob.cons_h(H_cons, x0)
    @test all(H_cons .≈ H_cons_expected)
    @test all(nnz.(H_cons) .== [4, 2])  # Check sparsity

    lag_H_expected = sparse(
        [1, 2, 3, 2, 3], [1, 2, 2, 3, 3], [6.0, 100.5, -199.0, -199.0, 204.5], 3, 3)
    σ = 1.0
    λ = [1.0, 2.0]
    lag_H = similar(optprob.lag_hess_prototype, Float64)
    optprob.lag_h(lag_H, x0, σ, λ)
    @test lag_H ≈ lag_H_expected
    @test nnz(lag_H) == 5

    optf = OptimizationFunction(sparse_objective,
        AutoSparse(DifferentiationInterface.SecondOrder(
            ADTypes.AutoForwardDiff(), ADTypes.AutoZygote())),
        cons = sparse_constraints)

    # Instantiate the optimization problem
    optprob = OptimizationBase.instantiate_function(optf, x0,
        AutoSparse(DifferentiationInterface.SecondOrder(
            ADTypes.AutoForwardDiff(), ADTypes.AutoZygote())),
        nothing, 2, g = true, h = true, cons_j = true, cons_h = true, lag_h = true)
    # Test gradient
    G = zeros(3)
    optprob.grad(G, x0)
    @test G ≈ [1.0, -50.0, 50.0]

    # Test Hessian
    H_expected = sparse(
        [1, 2, 2, 3, 3], [1, 2, 3, 2, 3], [2.0, 100.0, -200.0, -200.0, 200.0], 3, 3)
    H = similar(optprob.hess_prototype, Float64)
    optprob.hess(H, x0)
    @test H ≈ H_expected
    @test nnz(H) == 5  # Check sparsity

    # Test constraints
    res = zeros(2)
    optprob.cons(res, x0)
    @test res ≈ [0.0625, -0.5]

    # Test constraint Jacobian
    J_expected = sparse([1, 1, 1, 2, 2], [1, 2, 3, 1, 3], [1.0, 1.25, 0.25, 1.0, 1.0], 2, 3)
    J = similar(optprob.cons_jac_prototype, Float64)
    optprob.cons_j(J, x0)
    @test J ≈ J_expected
    @test nnz(J) == 5  # Check sparsity

    # Test constraint Hessians
    H_cons_expected = [sparse([2, 2, 3, 3], [2, 3, 2, 3], [0.5, 1.0, 1.0, 0.5], 3, 3),
        sparse([1, 3], [1, 3], [2.0, 2.0], 3, 3)]
    H_cons = [similar(h, Float64) for h in optprob.cons_hess_prototype]
    optprob.cons_h(H_cons, x0)
    @test all(H_cons .≈ H_cons_expected)
    @test all(nnz.(H_cons) .== [4, 2])  # Check sparsity

    lag_H_expected = sparse(
        [1, 2, 3, 2, 3], [1, 2, 2, 3, 3], [6.0, 100.5, -199.0, -199.0, 204.5], 3, 3)
    σ = 1.0
    λ = [1.0, 2.0]
    lag_H = similar(optprob.lag_hess_prototype, Float64)
    optprob.lag_h(lag_H, x0, σ, λ)
    @test lag_H ≈ lag_H_expected
    @test nnz(lag_H) == 5
end

@testset "OOP" begin
    cons = (x, p) -> [x[1]^2 + x[2]^2]
    optf = OptimizationFunction{false}(rosenbrock,
        OptimizationBase.AutoEnzyme(),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(
        optf, x0, OptimizationBase.AutoEnzyme(),
        nothing, 1, g = true, h = true, cons_j = true, cons_h = true)

    @test optprob.grad(x0) == G1
    @test optprob.hess(x0) == H1

    @test optprob.cons(x0) == [0.0]

    @test optprob.cons_j([5.0, 3.0]) == [10.0, 6.0]

    @test optprob.cons_h(x0) == [[2.0 0.0; 0.0 2.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2, x[2] * sin(x[1]) - x[1]]
    optf = OptimizationFunction{false}(rosenbrock,
        OptimizationBase.AutoEnzyme(),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(
        optf, x0, OptimizationBase.AutoEnzyme(),
        nothing, 2, g = true, h = true, cons_j = true, cons_h = true)

    @test optprob.grad(x0) == G1
    @test optprob.hess(x0) == H1
    @test optprob.cons(x0) == [0.0, 0.0]
    @test optprob.cons_j([5.0, 3.0])≈[10.0 6.0; -0.149013 -0.958924] rtol=1e-6
    @test optprob.cons_h(x0) == [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2]
    optf = OptimizationFunction{false}(rosenbrock,
        OptimizationBase.AutoFiniteDiff(),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(optf, x0,
        OptimizationBase.AutoFiniteDiff(),
        nothing, 1, g = true, h = true, cons_j = true, cons_h = true)

    @test optprob.grad(x0)≈G1 rtol=1e-6
    @test optprob.hess(x0)≈H1 rtol=1e-6

    @test optprob.cons(x0) == [0.0]

    @test optprob.cons_j([5.0, 3.0])≈[10.0, 6.0] rtol=1e-6

    @test optprob.cons_h(x0) ≈ [[2.0 0.0; 0.0 2.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2, x[2] * sin(x[1]) - x[1]]
    optf = OptimizationFunction{false}(rosenbrock,
        OptimizationBase.AutoFiniteDiff(),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(optf, x0,
        OptimizationBase.AutoFiniteDiff(),
        nothing, 2, g = true, h = true, cons_j = true, cons_h = true)

    @test optprob.grad(x0)≈G1 rtol=1e-6
    @test optprob.hess(x0)≈H1 rtol=1e-6
    @test optprob.cons(x0) == [0.0, 0.0]
    @test optprob.cons_j([5.0, 3.0])≈[10.0 6.0; -0.149013 -0.958924] rtol=1e-6
    @test optprob.cons_h(x0) ≈ [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2]
    optf = OptimizationFunction{false}(rosenbrock,
        OptimizationBase.AutoForwardDiff(),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(optf, x0,
        OptimizationBase.AutoForwardDiff(),
        nothing, 1, g = true, h = true, cons_j = true, cons_h = true)

    @test optprob.grad(x0) == G1
    @test optprob.hess(x0) == H1

    @test optprob.cons(x0) == [0.0]

    @test optprob.cons_j([5.0, 3.0]) == [10.0, 6.0]

    @test optprob.cons_h(x0) == [[2.0 0.0; 0.0 2.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2, x[2] * sin(x[1]) - x[1]]
    optf = OptimizationFunction{false}(rosenbrock,
        OptimizationBase.AutoForwardDiff(),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(optf, x0,
        OptimizationBase.AutoForwardDiff(),
        nothing, 2, g = true, h = true, cons_j = true, cons_h = true)

    @test optprob.grad(x0) == G1
    @test optprob.hess(x0) == H1
    @test optprob.cons(x0) == [0.0, 0.0]
    @test optprob.cons_j([5.0, 3.0])≈[10.0 6.0; -0.149013 -0.958924] rtol=1e-6
    @test optprob.cons_h(x0) == [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2]
    optf = OptimizationFunction{false}(rosenbrock,
        OptimizationBase.AutoReverseDiff(),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(optf, x0,
        OptimizationBase.AutoReverseDiff(),
        nothing, 1, g = true, h = true, cons_j = true, cons_h = true)

    @test optprob.grad(x0) == G1
    @test optprob.hess(x0) == H1

    @test optprob.cons(x0) == [0.0]

    @test optprob.cons_j([5.0, 3.0]) == [10.0, 6.0]

    @test optprob.cons_h(x0) == [[2.0 0.0; 0.0 2.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2, x[2] * sin(x[1]) - x[1]]
    optf = OptimizationFunction{false}(rosenbrock,
        OptimizationBase.AutoReverseDiff(),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(optf, x0,
        OptimizationBase.AutoReverseDiff(),
        nothing, 2, g = true, h = true, cons_j = true, cons_h = true)

    @test optprob.grad(x0) == G1
    @test optprob.hess(x0) == H1
    @test optprob.cons(x0) == [0.0, 0.0]
    @test optprob.cons_j([5.0, 3.0])≈[10.0 6.0; -0.149013 -0.958924] rtol=1e-6
    @test optprob.cons_h(x0) == [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2]
    optf = OptimizationFunction{false}(rosenbrock,
        OptimizationBase.AutoReverseDiff(true),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(optf, x0,
        OptimizationBase.AutoReverseDiff(true),
        nothing, 1, g = true, h = true, cons_j = true, cons_h = true)

    @test optprob.grad(x0) == G1
    @test optprob.hess(x0) == H1

    @test optprob.cons(x0) == [0.0]

    @test optprob.cons_j([5.0, 3.0]) == [10.0, 6.0]

    @test optprob.cons_h(x0) == [[2.0 0.0; 0.0 2.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2, x[2] * sin(x[1]) - x[1]]
    optf = OptimizationFunction{false}(rosenbrock,
        OptimizationBase.AutoReverseDiff(true),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(optf, x0,
        OptimizationBase.AutoReverseDiff(true),
        nothing, 2, g = true, h = true, cons_j = true, cons_h = true)

    @test optprob.grad(x0) == G1
    @test optprob.hess(x0) == H1
    @test optprob.cons(x0) == [0.0, 0.0]
    @test optprob.cons_j([5.0, 3.0])≈[10.0 6.0; -0.149013 -0.958924] rtol=1e-6
    @test optprob.cons_h(x0) == [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2]
    optf = OptimizationFunction{false}(rosenbrock,
        OptimizationBase.AutoSparseForwardDiff(),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(optf, x0,
        OptimizationBase.AutoSparseForwardDiff(),
        nothing, 1, g = true, h = true, cons_j = true, cons_h = true)

    @test optprob.grad(x0) == G1
    @test Array(optprob.hess(x0)) ≈ H1

    @test optprob.cons(x0) == [0.0]

    @test optprob.cons_j([5.0, 3.0]) == [10.0, 6.0]

    @test optprob.cons_h(x0) == [[2.0 0.0; 0.0 2.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2, x[2] * sin(x[1]) - x[1]]
    optf = OptimizationFunction{false}(rosenbrock,
        OptimizationBase.AutoSparseForwardDiff(),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(optf, x0,
        OptimizationBase.AutoSparseForwardDiff(),
        nothing, 2, g = true, h = true, cons_j = true, cons_h = true)

    @test optprob.grad(x0) == G1
    @test Array(optprob.hess(x0)) ≈ H1
    @test optprob.cons(x0) == [0.0, 0.0]
    @test Array(optprob.cons_j([5.0, 3.0]))≈[10.0 6.0; -0.149013 -0.958924] rtol=1e-6
    @test Array.(optprob.cons_h(x0)) ≈ [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2]
    optf = OptimizationFunction{false}(rosenbrock,
        OptimizationBase.AutoSparseFiniteDiff(),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(optf, x0,
        OptimizationBase.AutoSparseFiniteDiff(),
        nothing, 1, g = true, h = true, cons_j = true, cons_h = true)

    @test optprob.grad(x0)≈G1 rtol=1e-4
    @test Array(optprob.hess(x0)) ≈ H1

    @test optprob.cons(x0) == [0.0]

    @test optprob.cons_j([5.0, 3.0]) ≈ [10.0, 6.0]

    @test optprob.cons_h(x0) == [[2.0 0.0; 0.0 2.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2, x[2] * sin(x[1]) - x[1]]
    optf = OptimizationFunction{false}(rosenbrock,
        OptimizationBase.AutoSparseFiniteDiff(),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(optf, x0,
        OptimizationBase.AutoSparseForwardDiff(),
        nothing, 2, g = true, h = true, cons_j = true, cons_h = true)

    @test optprob.grad(x0) == G1
    @test Array(optprob.hess(x0)) ≈ H1
    @test optprob.cons(x0) == [0.0, 0.0]
    @test Array(optprob.cons_j([5.0, 3.0]))≈[10.0 6.0; -0.149013 -0.958924] rtol=1e-6
    @test Array.(optprob.cons_h(x0)) ≈ [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2]
    optf = OptimizationFunction{false}(rosenbrock,
        OptimizationBase.AutoSparseReverseDiff(),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(optf, x0,
        OptimizationBase.AutoSparseReverseDiff(),
        nothing, 1, g = true, h = true, cons_j = true, cons_h = true)

    @test optprob.grad(x0) == G1
    @test optprob.hess(x0) == H1

    @test optprob.cons(x0) == [0.0]

    @test optprob.cons_j([5.0, 3.0]) == [10.0, 6.0]

    @test optprob.cons_h(x0) == [[2.0 0.0; 0.0 2.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2, x[2] * sin(x[1]) - x[1]]
    optf = OptimizationFunction{false}(rosenbrock,
        OptimizationBase.AutoSparseReverseDiff(),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(optf, x0,
        OptimizationBase.AutoSparseReverseDiff(),
        nothing, 2, g = true, h = true, cons_j = true, cons_h = true)

    @test optprob.grad(x0) == G1
    @test Array(optprob.hess(x0)) ≈ H1
    @test optprob.cons(x0) == [0.0, 0.0]
    @test Array(optprob.cons_j([5.0, 3.0]))≈[10.0 6.0; -0.149013 -0.958924] rtol=1e-6
    @test Array.(optprob.cons_h(x0)) ≈ [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2]
    optf = OptimizationFunction{false}(rosenbrock,
        OptimizationBase.AutoSparseReverseDiff(true),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(optf, x0,
        OptimizationBase.AutoSparseReverseDiff(true),
        nothing, 1, g = true, h = true, cons_j = true, cons_h = true)

    @test optprob.grad(x0) == G1
    @test optprob.hess(x0) == H1
    @test optprob.cons(x0) == [0.0]

    @test optprob.cons_j([5.0, 3.0]) == [10.0, 6.0]

    @test optprob.cons_h(x0) == [[2.0 0.0; 0.0 2.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2, x[2] * sin(x[1]) - x[1]]
    optf = OptimizationFunction{false}(rosenbrock,
        OptimizationBase.AutoSparseReverseDiff(true),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(optf, x0,
        OptimizationBase.AutoSparseReverseDiff(true),
        nothing, 2, g = true, h = true, cons_j = true, cons_h = true)

    @test optprob.grad(x0) == G1
    @test Array(optprob.hess(x0)) ≈ H1
    @test optprob.cons(x0) == [0.0, 0.0]
    @test Array(optprob.cons_j([5.0, 3.0]))≈[10.0 6.0; -0.149013 -0.958924] rtol=1e-6
    @test Array.(optprob.cons_h(x0)) ≈ [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2]
    optf = OptimizationFunction{false}(rosenbrock,
        AutoZygote(),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(
        optf, x0, AutoZygote(),
        nothing, 1, g = true, h = true, cons_j = true, cons_h = true)

    @test optprob.grad(x0) == G1
    @test optprob.hess(x0) == H1
    @test optprob.cons(x0) == [0.0]

    @test optprob.cons_j([5.0, 3.0]) == [10.0, 6.0]

    @test optprob.cons_h(x0) == [[2.0 0.0; 0.0 2.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2, x[2] * sin(x[1]) - x[1]]
    optf = OptimizationFunction{false}(rosenbrock,
        AutoZygote(),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(
        optf, x0, AutoZygote(),
        nothing, 2, g = true, h = true, cons_j = true, cons_h = true)

    @test optprob.grad(x0) == G1
    @test Array(optprob.hess(x0)) ≈ H1
    @test optprob.cons(x0) == [0.0, 0.0]
    @test optprob.cons_j([5.0, 3.0])≈[10.0 6.0; -0.149013 -0.958924] rtol=1e-6
    @test Array.(optprob.cons_h(x0)) ≈ [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]
end

using MLUtils

@testset "Stochastic gradient" begin
    x0 = rand(10000)
    y0 = sin.(x0)
    data = MLUtils.DataLoader((x0, y0), batchsize = 100)

    function loss(coeffs, data)
        ypred = [evalpoly(data[1][i], coeffs) for i in eachindex(data[1])]
        return sum(abs2, ypred .- data[2])
    end

    optf = OptimizationFunction(loss, AutoForwardDiff())
    optf = OptimizationBase.instantiate_function(
        optf, rand(3), AutoForwardDiff(), iterate(data)[1], g = true, fg = true)
    G0 = zeros(3)
    optf.grad(G0, ones(3), (x0, y0))
    stochgrads = []
    i = 0
    for (x, y) in data
        G = zeros(3)
        optf.grad(G, ones(3), (x, y))
        push!(stochgrads, copy(G))
        G1 = zeros(3)
        optf.fg(G1, ones(3), (x, y))
        @test G≈G1 rtol=1e-6
    end
    @test G0≈sum(stochgrads) rtol=1e-1

    optf = OptimizationFunction(loss, AutoReverseDiff())
    optf = OptimizationBase.instantiate_function(
        optf, rand(3), AutoReverseDiff(), iterate(data)[1], g = true, fg = true)
    G0 = zeros(3)
    optf.grad(G0, ones(3), (x0, y0))
    stochgrads = []
    for (x, y) in data
        G = zeros(3)
        optf.grad(G, ones(3), (x, y))
        push!(stochgrads, copy(G))
        G1 = zeros(3)
        optf.fg(G1, ones(3), (x, y))
        @test G≈G1 rtol=1e-6
    end
    @test G0≈sum(stochgrads) rtol=1e-1

    optf = OptimizationFunction(loss, AutoZygote())
    optf = OptimizationBase.instantiate_function(
        optf, rand(3), AutoZygote(), iterate(data)[1], g = true, fg = true)
    G0 = zeros(3)
    optf.grad(G0, ones(3), (x0, y0))
    stochgrads = []
    for (x, y) in data
        G = zeros(3)
        optf.grad(G, ones(3), (x, y))
        push!(stochgrads, copy(G))
        G1 = zeros(3)
        optf.fg(G1, ones(3), (x, y))
        @test G≈G1 rtol=1e-6
    end
    @test G0≈sum(stochgrads) rtol=1e-1

    optf = OptimizationFunction(loss, AutoEnzyme())
    optf = OptimizationBase.instantiate_function(
        optf, rand(3), AutoEnzyme(mode = set_runtime_activity(Reverse)),
        iterate(data)[1], g = true, fg = true)
    G0 = zeros(3)
    optf.grad(G0, ones(3), (x0, y0))
    stochgrads = []
    for (x, y) in data
        G = zeros(3)
        optf.grad(G, ones(3), (x, y))
        push!(stochgrads, copy(G))
        G1 = zeros(3)
        optf.fg(G1, ones(3), (x, y))
        @test G≈G1 rtol=1e-6
    end
    @test G0≈sum(stochgrads) rtol=1e-1
end
