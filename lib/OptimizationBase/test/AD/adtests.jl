using OptimizationBase, Test, DifferentiationInterface, LinearAlgebra, SparseArrays, Symbolics
using ADTypes, ForwardDiff, Zygote, ReverseDiff, FiniteDiff, Tracker
using ModelingToolkit, Enzyme, Random, ComponentArrays, JLArrays

x0 = zeros(2)
rosenbrock(x, p = nothing) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
component_rosenbrock(x, p = nothing) = (1 - x.x[1])^2 + 100 * (x.x[2] - x.x[1]^2)^2
jlarray_constant(x::JLArray, p = nothing) = zero(eltype(x))
l1 = rosenbrock(x0)

function g!(G, x)
    G[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    return G[2] = 200.0 * (x[2] - x[1]^2)
end

function h!(H, x)
    H[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
    H[1, 2] = -400.0 * x[1]
    H[2, 1] = -400.0 * x[1]
    return H[2, 2] = 200.0
end

G1 = Array{Float64}(undef, 2)
G2 = Array{Float64}(undef, 2)
H1 = Array{Float64}(undef, 2, 2)
H2 = Array{Float64}(undef, 2, 2)

g!(G1, x0)
h!(H1, x0)

cons = (res, x, p) -> (res[1] = x[1]^2 + x[2]^2; return nothing)
optf = OptimizationFunction(rosenbrock, OptimizationBase.AutoSymbolics(), cons = cons)
optprob = OptimizationBase.instantiate_function(
    optf, x0,
    OptimizationBase.AutoSymbolics(),
    nothing, 1, g = true, h = true, cons_j = true, cons_h = true
)
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
optf = OptimizationFunction(
    rosenbrock,
    OptimizationBase.AutoSymbolics(),
    cons = con2_c
)
optprob = OptimizationBase.instantiate_function(
    optf, x0,
    OptimizationBase.AutoSymbolics(),
    nothing, 2, g = true, h = true, cons_j = true, cons_h = true
)
optprob.grad(G2, x0)
@test G1 == G2
optprob.hess(H2, x0)
@test H1 == H2
res = Array{Float64}(undef, 2)
optprob.cons(res, x0)
@test res == [0.0, 0.0]
J = Array{Float64}(undef, 2, 2)
optprob.cons_j(J, [5.0, 3.0])
@test all(isapprox(J, [10.0 6.0; -0.149013 -0.958924]; rtol = 1.0e-3))
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
        cons_jvp = true, lag_h = true
    )
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
    @test H4 ≈ σ * H2 + μ[1] * H3[1] rtol = 1.0e-6

    @testset "ComponentVector container preservation" begin
        θ = ComponentVector(x = zeros(2))
        optprob_abstractvector = OptimizationBase.instantiate_function(
            OptimizationFunction(component_rosenbrock, OptimizationBase.AutoEnzyme()), θ,
            OptimizationBase.AutoEnzyme(), nothing, 0, h = true, fgh = true
        )
        G_abstractvector = Enzyme.make_zero(θ)
        H_abstractvector = zeros(2, 2)
        optprob_abstractvector.hess(H_abstractvector, θ)
        @test H1 == H_abstractvector
        optprob_abstractvector.fgh(G_abstractvector, H_abstractvector, θ)
        @test typeof(G_abstractvector) === typeof(θ)
        @test G1 == collect(G_abstractvector)
        @test H1 == H_abstractvector
    end

    @testset "JLArray container preservation" begin
        x_jlarray = [1.0, 2.0]
        θ = JLArray(x_jlarray)
        optprob_abstractvector = OptimizationBase.instantiate_function(
            OptimizationFunction(jlarray_constant, OptimizationBase.AutoEnzyme()), θ,
            OptimizationBase.AutoEnzyme(), nothing, 0, h = true, fgh = true
        )
        G_abstractvector = zero(θ)
        fill!(G_abstractvector, 3)
        H_abstractvector = fill(3.0, 2, 2)
        optprob_abstractvector.hess(H_abstractvector, θ)
        @test iszero(H_abstractvector)
        @test x_jlarray == collect(θ)
        fill!(G_abstractvector, 3)
        fill!(H_abstractvector, 3)
        optprob_abstractvector.fgh(G_abstractvector, H_abstractvector, θ)
        @test typeof(G_abstractvector) === typeof(θ)
        @test iszero(collect(G_abstractvector))
        @test iszero(H_abstractvector)
        @test x_jlarray == collect(θ)
    end

    G2 = Array{Float64}(undef, 2)
    H2 = Array{Float64}(undef, 2, 2)

    optf = OptimizationFunction(rosenbrock, OptimizationBase.AutoForwardDiff(), cons = cons)
    optprob = OptimizationBase.instantiate_function(
        optf, x0, OptimizationBase.AutoForwardDiff(),
        nothing, 1, g = true, h = true, hv = true,
        cons_j = true, cons_h = true, cons_vjp = true,
        cons_jvp = true, lag_h = true
    )
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
    @test H4 ≈ σ * H2 + μ[1] * H3[1] rtol = 1.0e-6

    # Test that the AD-generated lag_hess_prototype has correct dimensions
    @test !isnothing(optprob.lag_hess_prototype)
    @test size(optprob.lag_hess_prototype) == (length(x0), length(x0))  # Should be n×n, not num_cons×n

    # Test that we can actually use it as a buffer
    if !isnothing(optprob.lag_hess_prototype)
        H_proto = similar(optprob.lag_hess_prototype, Float64)
        optprob.lag_h(H_proto, x0, σ, μ)
        @test H_proto ≈ σ * H2 + μ[1] * H3[1] rtol = 1.0e-6
    end

    G2 = Array{Float64}(undef, 2)
    H2 = Array{Float64}(undef, 2, 2)

    optf = OptimizationFunction(rosenbrock, OptimizationBase.AutoReverseDiff(), cons = cons)
    optprob = OptimizationBase.instantiate_function(
        optf, x0, OptimizationBase.AutoReverseDiff(),
        nothing, 1, g = true, h = true, hv = true,
        cons_j = true, cons_h = true, cons_vjp = true,
        cons_jvp = true, lag_h = true
    )
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
    @test H4 ≈ σ * H2 + μ[1] * H3[1] rtol = 1.0e-6

    G2 = Array{Float64}(undef, 2)
    H2 = Array{Float64}(undef, 2, 2)

    optf = OptimizationFunction(
        rosenbrock, OptimizationBase.AutoReverseDiff(; compile = true), cons = cons
    )
    optprob = OptimizationBase.instantiate_function(
        optf, x0, OptimizationBase.AutoReverseDiff(; compile = true),
        nothing, 1, g = true, h = true, hv = true,
        cons_j = true, cons_h = true, cons_vjp = true,
        cons_jvp = true, lag_h = true
    )
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
    @test H4 ≈ σ * H2 + μ[1] * H3[1] rtol = 1.0e-6

    G2 = Array{Float64}(undef, 2)
    H2 = Array{Float64}(undef, 2, 2)

    optf = OptimizationFunction(
        rosenbrock, AutoZygote(), cons = cons
    )
    optprob = OptimizationBase.instantiate_function(
        optf, x0, AutoZygote(),
        nothing, 1, g = true, h = true, hv = true,
        cons_j = true, cons_h = true, cons_vjp = true,
        cons_jvp = true, lag_h = true
    )
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
    @test H4 ≈ σ * H2 + μ[1] * H3[1] rtol = 1.0e-6

    # Test that the AD-generated lag_hess_prototype has correct dimensions
    @test !isnothing(optprob.lag_hess_prototype)
    @test size(optprob.lag_hess_prototype) == (length(x0), length(x0))  # Should be n×n, not num_cons×n

    # Test that we can actually use it as a buffer (this would fail with the bug)
    if !isnothing(optprob.lag_hess_prototype)
        H_proto = similar(optprob.lag_hess_prototype, Float64)
        optprob.lag_h(H_proto, x0, σ, μ)
        @test H_proto ≈ σ * H2 + μ[1] * H3[1] rtol = 1.0e-6
    end

    G2 = Array{Float64}(undef, 2)
    H2 = Array{Float64}(undef, 2, 2)

    optf = OptimizationFunction(
        rosenbrock,
        DifferentiationInterface.SecondOrder(
            ADTypes.AutoFiniteDiff(), ADTypes.AutoReverseDiff()
        ),
        cons = cons
    )
    optprob = OptimizationBase.instantiate_function(
        optf, x0,
        DifferentiationInterface.SecondOrder(
            ADTypes.AutoFiniteDiff(), ADTypes.AutoReverseDiff()
        ),
        nothing, 1, g = true, h = true, hv = true,
        cons_j = true, cons_h = true, cons_vjp = true,
        cons_jvp = true, lag_h = true
    )
    optprob.grad(G2, x0)
    @test G1 ≈ G2 rtol = 1.0e-5
    optprob.hess(H2, x0)
    @test H1 ≈ H2 rtol = 1.0e-5
    Hv = Array{Float64}(undef, 2)
    optprob.hv(Hv, x0, [1.0, 1.0])
    @test Hv ≈ [2.0, 200.0] rtol = 1.0e-5
    res = Array{Float64}(undef, 1)
    optprob.cons(res, x0)
    @test res ≈ [0.0]
    J = Array{Float64}(undef, 1, 2)
    optprob.cons_j(J, [5.0, 3.0])
    @test J ≈ [10.0 6.0] rtol = 1.0e-5
    vJ = Array{Float64}(undef, 2)
    optprob.cons_vjp(vJ, [5.0, 3.0], [1.0])
    @test vJ ≈ [10.0, 6.0] rtol = 1.0e-5
    Jv = Array{Float64}(undef, 1)
    optprob.cons_jvp(Jv, [5.0, 3.0], [0.5, 0.5])
    @test Jv ≈ [8.0] rtol = 1.0e-5
    H3 = [Array{Float64}(undef, 2, 2)]
    optprob.cons_h(H3, x0)
    @test H3 ≈ [[2.0 0.0; 0.0 2.0]] rtol = 1.0e-5
    Random.seed!(123)
    H4 = Array{Float64}(undef, 2, 2)
    μ = randn(1)
    σ = rand()
    optprob.lag_h(H4, x0, σ, μ)
    @test H4 ≈ σ * H2 + μ[1] * H3[1] rtol = 1.0e-6
end

@testset "two constraints tests" begin
    G2 = Array{Float64}(undef, 2)
    H2 = Array{Float64}(undef, 2, 2)
    optf = OptimizationFunction(rosenbrock, OptimizationBase.AutoEnzyme(), cons = con2_c)
    optprob = OptimizationBase.instantiate_function(
        optf, x0, OptimizationBase.AutoEnzyme(),
        nothing, 2, g = true, h = true, hv = true,
        cons_j = true, cons_h = true, cons_vjp = true,
        cons_jvp = true, lag_h = true
    )
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
    @test all(isapprox(J, [10.0 6.0; -0.149013 -0.958924]; rtol = 1.0e-3))
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
    @test H4 ≈ σ * H1 + sum(μ .* H3) rtol = 1.0e-6

    G2 = Array{Float64}(undef, 2)
    H2 = Array{Float64}(undef, 2, 2)

    optf = OptimizationFunction(
        rosenbrock, OptimizationBase.AutoReverseDiff(), cons = con2_c
    )
    optprob = OptimizationBase.instantiate_function(
        optf, x0,
        OptimizationBase.AutoReverseDiff(),
        nothing, 2, g = true, h = true, hv = true,
        cons_j = true, cons_h = true, cons_vjp = true,
        cons_jvp = true, lag_h = true
    )
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
    @test all(isapprox(J, [10.0 6.0; -0.149013 -0.958924]; rtol = 1.0e-3))
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
    @test H4 ≈ σ * H1 + sum(μ .* H3) rtol = 1.0e-6

    G2 = Array{Float64}(undef, 2)
    H2 = Array{Float64}(undef, 2, 2)

    optf = OptimizationFunction(
        rosenbrock, OptimizationBase.AutoReverseDiff(; compile = true), cons = con2_c
    )
    optprob = OptimizationBase.instantiate_function(
        optf, x0,
        OptimizationBase.AutoReverseDiff(; compile = true),
        nothing, 2, g = true, h = true, hv = true,
        cons_j = true, cons_h = true, cons_vjp = true,
        cons_jvp = true, lag_h = true
    )
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
    @test all(isapprox(J, [10.0 6.0; -0.149013 -0.958924]; rtol = 1.0e-3))
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
    @test H4 ≈ σ * H1 + sum(μ .* H3) rtol = 1.0e-6

    G2 = Array{Float64}(undef, 2)
    H2 = Array{Float64}(undef, 2, 2)

    optf = OptimizationFunction(
        rosenbrock, OptimizationBase.AutoForwardDiff(), cons = con2_c
    )
    optprob = OptimizationBase.instantiate_function(
        optf, x0,
        OptimizationBase.AutoReverseDiff(; compile = true),
        nothing, 2, g = true, h = true, hv = true,
        cons_j = true, cons_h = true, cons_vjp = true,
        cons_jvp = true, lag_h = true
    )
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
    @test all(isapprox(J, [10.0 6.0; -0.149013 -0.958924]; rtol = 1.0e-3))
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
    @test H4 ≈ σ * H1 + sum(μ .* H3) rtol = 1.0e-6

    G2 = Array{Float64}(undef, 2)
    H2 = Array{Float64}(undef, 2, 2)

    optf = OptimizationFunction(
        rosenbrock, AutoZygote(), cons = con2_c
    )
    optprob = OptimizationBase.instantiate_function(
        optf, x0, AutoZygote(),
        nothing, 2, g = true, h = true, hv = true,
        cons_j = true, cons_h = true, cons_vjp = true,
        cons_jvp = true, lag_h = true
    )
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
    @test all(isapprox(J, [10.0 6.0; -0.149013 -0.958924]; rtol = 1.0e-3))
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
    @test H4 ≈ σ * H1 + sum(μ .* H3) rtol = 1.0e-6

    # Test that the AD-generated lag_hess_prototype has correct dimensions
    @test !isnothing(optprob.lag_hess_prototype)
    @test size(optprob.lag_hess_prototype) == (length(x0), length(x0))  # Should be n×n, not num_cons×n

    # Test that we can actually use it as a buffer (this would fail with the bug)
    if !isnothing(optprob.lag_hess_prototype)
        H_proto = similar(optprob.lag_hess_prototype, Float64)
        optprob.lag_h(H_proto, x0, σ, μ)
        @test H_proto ≈ σ * H1 + sum(μ .* H3) rtol = 1.0e-6
    end

    G2 = Array{Float64}(undef, 2)
    H2 = Array{Float64}(undef, 2, 2)

    optf = OptimizationFunction(
        rosenbrock, DifferentiationInterface.SecondOrder(
            ADTypes.AutoFiniteDiff(), ADTypes.AutoReverseDiff()
        ),
        cons = con2_c
    )
    optprob = OptimizationBase.instantiate_function(
        optf, x0,
        DifferentiationInterface.SecondOrder(
            ADTypes.AutoFiniteDiff(), ADTypes.AutoReverseDiff()
        ),
        nothing, 2, g = true, h = true, hv = true,
        cons_j = true, cons_h = true, cons_vjp = true,
        cons_jvp = true, lag_h = true
    )
    optprob.grad(G2, x0)
    @test G1 ≈ G2 rtol = 1.0e-5
    optprob.hess(H2, x0)
    @test H1 ≈ H2 rtol = 1.0e-5
    Hv = Array{Float64}(undef, 2)
    optprob.hv(Hv, x0, [1.0, 1.0])
    @test Hv ≈ [2.0, 200.0] rtol = 1.0e-5
    res = Array{Float64}(undef, 2)
    optprob.cons(res, x0)
    @test res ≈ [0.0, 0.0]
    J = Array{Float64}(undef, 2, 2)
    optprob.cons_j(J, [5.0, 3.0])
    @test all(isapprox(J, [10.0 6.0; -0.149013 -0.958924]; rtol = 1.0e-3))
    vJ = Array{Float64}(undef, 2)
    optprob.cons_vjp(vJ, [5.0, 3.0], [1.0, 1.0])
    @test vJ ≈ sum(J, dims = 1)[:] rtol = 1.0e-5
    Jv = Array{Float64}(undef, 2)
    optprob.cons_jvp(Jv, [5.0, 3.0], [0.5, 0.5])
    @test Jv ≈ 0.5 * sum(J, dims = 2)[:] rtol = 1.0e-5
    H3 = [Array{Float64}(undef, 2, 2), Array{Float64}(undef, 2, 2)]
    optprob.cons_h(H3, x0)
    @test H3 ≈ [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]] rtol = 1.0e-5
    H4 = Array{Float64}(undef, 2, 2)
    μ = randn(2)
    σ = rand()
    optprob.lag_h(H4, x0, σ, μ)
    @test H4 ≈ σ * H1 + sum(μ .* H3) rtol = 1.0e-6
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
    optf = OptimizationFunction(
        sparse_objective, AutoSparse(OptimizationBase.AutoForwardDiff()),
        cons = sparse_constraints
    )

    # Instantiate the optimization problem
    optprob = OptimizationBase.instantiate_function(
        optf, x0,
        AutoSparse(OptimizationBase.AutoForwardDiff()),
        nothing, 2, g = true, h = true, cons_j = true, cons_h = true, lag_h = true
    )
    # Test gradient
    G = zeros(3)
    optprob.grad(G, x0)
    @test G ≈ [1.0, -50.0, 50.0]

    # Test Hessian
    H_expected = sparse(
        [1, 2, 2, 3, 3], [1, 2, 3, 2, 3], [2.0, 100.0, -200.0, -200.0, 200.0], 3, 3
    )
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
    H_cons_expected = [
        sparse([2, 2, 3, 3], [2, 3, 2, 3], [0.5, 1.0, 1.0, 0.5], 3, 3),
        sparse([1, 3], [1, 3], [2.0, 2.0], 3, 3),
    ]
    H_cons = [similar(h, Float64) for h in optprob.cons_hess_prototype]
    optprob.cons_h(H_cons, x0)
    @test all(H_cons .≈ H_cons_expected)
    @test all(nnz.(H_cons) .== [4, 2])  # Check sparsity

    lag_H_expected = sparse(
        [1, 2, 3, 2, 3], [1, 2, 2, 3, 3], [6.0, 100.5, -199.0, -199.0, 204.5], 3, 3
    )
    σ = 1.0
    λ = [1.0, 2.0]
    lag_H = similar(optprob.lag_hess_prototype, Float64)
    optprob.lag_h(lag_H, x0, σ, λ)
    @test lag_H ≈ lag_H_expected
    @test nnz(lag_H) == 5

    optf = OptimizationFunction(
        sparse_objective, AutoSparse(OptimizationBase.AutoReverseDiff()),
        cons = sparse_constraints
    )

    # Instantiate the optimization problem
    optprob = OptimizationBase.instantiate_function(
        optf, x0,
        AutoSparse(OptimizationBase.AutoForwardDiff()),
        nothing, 2, g = true, h = true, cons_j = true, cons_h = true, lag_h = true
    )
    # Test gradient
    G = zeros(3)
    optprob.grad(G, x0)
    @test G ≈ [1.0, -50.0, 50.0]

    # Test Hessian
    H_expected = sparse(
        [1, 2, 2, 3, 3], [1, 2, 3, 2, 3], [2.0, 100.0, -200.0, -200.0, 200.0], 3, 3
    )
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
    H_cons_expected = [
        sparse([2, 2, 3, 3], [2, 3, 2, 3], [0.5, 1.0, 1.0, 0.5], 3, 3),
        sparse([1, 3], [1, 3], [2.0, 2.0], 3, 3),
    ]
    H_cons = [similar(h, Float64) for h in optprob.cons_hess_prototype]
    optprob.cons_h(H_cons, x0)
    @test all(H_cons .≈ H_cons_expected)
    @test all(nnz.(H_cons) .== [4, 2])  # Check sparsity

    lag_H_expected = sparse(
        [1, 2, 3, 2, 3], [1, 2, 2, 3, 3], [6.0, 100.5, -199.0, -199.0, 204.5], 3, 3
    )
    σ = 1.0
    λ = [1.0, 2.0]
    lag_H = similar(optprob.lag_hess_prototype, Float64)
    optprob.lag_h(lag_H, x0, σ, λ)
    @test lag_H ≈ lag_H_expected
    @test nnz(lag_H) == 5

    optf = OptimizationFunction(
        sparse_objective, AutoSparse(OptimizationBase.AutoReverseDiff(; compile = true)),
        cons = sparse_constraints
    )

    # Instantiate the optimization problem
    optprob = OptimizationBase.instantiate_function(
        optf, x0,
        AutoSparse(OptimizationBase.AutoForwardDiff()),
        nothing, 2, g = true, h = true, cons_j = true, cons_h = true, lag_h = true
    )
    # Test gradient
    G = zeros(3)
    optprob.grad(G, x0)
    @test G ≈ [1.0, -50.0, 50.0]

    # Test Hessian
    H_expected = sparse(
        [1, 2, 2, 3, 3], [1, 2, 3, 2, 3], [2.0, 100.0, -200.0, -200.0, 200.0], 3, 3
    )
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
    H_cons_expected = [
        sparse([2, 2, 3, 3], [2, 3, 2, 3], [0.5, 1.0, 1.0, 0.5], 3, 3),
        sparse([1, 3], [1, 3], [2.0, 2.0], 3, 3),
    ]
    H_cons = [similar(h, Float64) for h in optprob.cons_hess_prototype]
    optprob.cons_h(H_cons, x0)
    @test all(H_cons .≈ H_cons_expected)
    @test all(nnz.(H_cons) .== [4, 2])  # Check sparsity

    lag_H_expected = sparse(
        [1, 2, 3, 2, 3], [1, 2, 2, 3, 3], [6.0, 100.5, -199.0, -199.0, 204.5], 3, 3
    )
    σ = 1.0
    λ = [1.0, 2.0]
    lag_H = similar(optprob.lag_hess_prototype, Float64)
    optprob.lag_h(lag_H, x0, σ, λ)
    @test lag_H ≈ lag_H_expected
    @test nnz(lag_H) == 5

    optf = OptimizationFunction(
        sparse_objective, AutoSparse(OptimizationBase.AutoFiniteDiff()),
        cons = sparse_constraints
    )

    # Instantiate the optimization problem
    optprob = OptimizationBase.instantiate_function(
        optf, x0,
        AutoSparse(OptimizationBase.AutoForwardDiff()),
        nothing, 2, g = true, h = true, cons_j = true, cons_h = true, lag_h = true
    )
    # Test gradient
    G = zeros(3)
    optprob.grad(G, x0)
    @test G ≈ [1.0, -50.0, 50.0]

    # Test Hessian
    H_expected = sparse(
        [1, 2, 2, 3, 3], [1, 2, 3, 2, 3], [2.0, 100.0, -200.0, -200.0, 200.0], 3, 3
    )
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
    H_cons_expected = [
        sparse([2, 2, 3, 3], [2, 3, 2, 3], [0.5, 1.0, 1.0, 0.5], 3, 3),
        sparse([1, 3], [1, 3], [2.0, 2.0], 3, 3),
    ]
    H_cons = [similar(h, Float64) for h in optprob.cons_hess_prototype]
    optprob.cons_h(H_cons, x0)
    @test all(H_cons .≈ H_cons_expected)
    @test all(nnz.(H_cons) .== [4, 2])  # Check sparsity

    lag_H_expected = sparse(
        [1, 2, 3, 2, 3], [1, 2, 2, 3, 3], [6.0, 100.5, -199.0, -199.0, 204.5], 3, 3
    )
    σ = 1.0
    λ = [1.0, 2.0]
    lag_H = similar(optprob.lag_hess_prototype, Float64)
    optprob.lag_h(lag_H, x0, σ, λ)
    @test lag_H ≈ lag_H_expected
    @test nnz(lag_H) == 5

    optf = OptimizationFunction(
        sparse_objective,
        AutoSparse(
            DifferentiationInterface.SecondOrder(
                ADTypes.AutoForwardDiff(), ADTypes.AutoZygote()
            )
        ),
        cons = sparse_constraints
    )

    # Instantiate the optimization problem
    optprob = OptimizationBase.instantiate_function(
        optf, x0,
        AutoSparse(
            DifferentiationInterface.SecondOrder(
                ADTypes.AutoForwardDiff(), ADTypes.AutoZygote()
            )
        ),
        nothing, 2, g = true, h = true, cons_j = true, cons_h = true, lag_h = true
    )
    # Test gradient
    G = zeros(3)
    optprob.grad(G, x0)
    @test G ≈ [1.0, -50.0, 50.0]

    # Test Hessian
    H_expected = sparse(
        [1, 2, 2, 3, 3], [1, 2, 3, 2, 3], [2.0, 100.0, -200.0, -200.0, 200.0], 3, 3
    )
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
    H_cons_expected = [
        sparse([2, 2, 3, 3], [2, 3, 2, 3], [0.5, 1.0, 1.0, 0.5], 3, 3),
        sparse([1, 3], [1, 3], [2.0, 2.0], 3, 3),
    ]
    H_cons = [similar(h, Float64) for h in optprob.cons_hess_prototype]
    optprob.cons_h(H_cons, x0)
    @test all(H_cons .≈ H_cons_expected)
    @test all(nnz.(H_cons) .== [4, 2])  # Check sparsity

    lag_H_expected = sparse(
        [1, 2, 3, 2, 3], [1, 2, 2, 3, 3], [6.0, 100.5, -199.0, -199.0, 204.5], 3, 3
    )
    σ = 1.0
    λ = [1.0, 2.0]
    lag_H = similar(optprob.lag_hess_prototype, Float64)
    optprob.lag_h(lag_H, x0, σ, λ)
    @test lag_H ≈ lag_H_expected
    @test nnz(lag_H) == 5
end

@testset "sparse constraint Jacobians use live parameters" begin
    objective(x, p) = sum(abs2, x)
    constraints(x, p) = [p[1] * x[1]]
    function constraints!(res, x, p)
        res[1] = p[1] * x[1]
        return nothing
    end
    constraint_jacobian(x, p) = sparse([1], [1], [p[1]], 1, 2)
    function constraint_jacobian!(J, x, p)
        J[1, 1] = p[1]
        return nothing
    end

    x = [1.0, 1.0]
    initial_p = [2.0]
    live_p = [3.0]

    @testset "in-place = $iip, analytic = $analytic" for iip in (false, true),
            analytic in (false, true)
        jacobian_kwargs = analytic ?
            (;
                cons_j = iip ? constraint_jacobian! : constraint_jacobian,
                cons_jac_prototype = sparse([1], [1], [1.0], 1, 2),
            ) : (;)
        optf = OptimizationFunction{iip}(
            objective, AutoSparse(AutoForwardDiff());
            cons = iip ? constraints! : constraints, jacobian_kwargs...
        )
        optprob = OptimizationBase.instantiate_function(
            optf, x, optf.adtype, initial_p, 1; cons_j = true
        )

        if iip
            J = similar(optprob.cons_jac_prototype, Float64)
            optprob.cons_j(J, x, live_p)
            @test vec(Array(J)) == [3.0, 0.0]
        else
            @test vec(Array(optprob.cons_j(x, live_p))) == [3.0, 0.0]
        end
    end
end

@testset "OOP" begin
    cons = (x, p) -> [x[1]^2 + x[2]^2]
    optf = OptimizationFunction{false}(
        rosenbrock,
        OptimizationBase.AutoEnzyme(),
        cons = cons
    )
    optprob = OptimizationBase.instantiate_function(
        optf, x0, OptimizationBase.AutoEnzyme(),
        nothing, 1, g = true, h = true, cons_j = true, cons_h = true
    )

    @test optprob.grad(x0) == G1
    @test optprob.hess(x0) == H1

    @test optprob.cons(x0) == [0.0]

    @test optprob.cons_j([5.0, 3.0]) == [10.0, 6.0]

    @test optprob.cons_h(x0) == [[2.0 0.0; 0.0 2.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2, x[2] * sin(x[1]) - x[1]]
    optf = OptimizationFunction{false}(
        rosenbrock,
        OptimizationBase.AutoEnzyme(),
        cons = cons
    )
    optprob = OptimizationBase.instantiate_function(
        optf, x0, OptimizationBase.AutoEnzyme(),
        nothing, 2, g = true, h = true, cons_j = true, cons_h = true
    )

    @test optprob.grad(x0) == G1
    @test optprob.hess(x0) == H1
    @test optprob.cons(x0) == [0.0, 0.0]
    @test optprob.cons_j([5.0, 3.0]) ≈ [10.0 6.0; -0.149013 -0.958924] rtol = 1.0e-6
    @test optprob.cons_h(x0) == [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2]
    optf = OptimizationFunction{false}(
        rosenbrock,
        OptimizationBase.AutoFiniteDiff(),
        cons = cons
    )
    optprob = OptimizationBase.instantiate_function(
        optf, x0,
        OptimizationBase.AutoFiniteDiff(),
        nothing, 1, g = true, h = true, cons_j = true, cons_h = true
    )

    @test optprob.grad(x0) ≈ G1 rtol = 1.0e-6
    @test optprob.hess(x0) ≈ H1 rtol = 1.0e-6

    @test optprob.cons(x0) == [0.0]

    @test optprob.cons_j([5.0, 3.0]) ≈ [10.0, 6.0] rtol = 1.0e-6

    @test optprob.cons_h(x0) ≈ [[2.0 0.0; 0.0 2.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2, x[2] * sin(x[1]) - x[1]]
    optf = OptimizationFunction{false}(
        rosenbrock,
        OptimizationBase.AutoFiniteDiff(),
        cons = cons
    )
    optprob = OptimizationBase.instantiate_function(
        optf, x0,
        OptimizationBase.AutoFiniteDiff(),
        nothing, 2, g = true, h = true, cons_j = true, cons_h = true
    )

    @test optprob.grad(x0) ≈ G1 rtol = 1.0e-6
    @test optprob.hess(x0) ≈ H1 rtol = 1.0e-6
    @test optprob.cons(x0) == [0.0, 0.0]
    @test optprob.cons_j([5.0, 3.0]) ≈ [10.0 6.0; -0.149013 -0.958924] rtol = 1.0e-6
    @test optprob.cons_h(x0) ≈ [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2]
    optf = OptimizationFunction{false}(
        rosenbrock,
        OptimizationBase.AutoForwardDiff(),
        cons = cons
    )
    optprob = OptimizationBase.instantiate_function(
        optf, x0,
        OptimizationBase.AutoForwardDiff(),
        nothing, 1, g = true, h = true, cons_j = true, cons_h = true
    )

    @test optprob.grad(x0) == G1
    @test optprob.hess(x0) == H1

    @test optprob.cons(x0) == [0.0]

    @test optprob.cons_j([5.0, 3.0]) == [10.0, 6.0]

    @test optprob.cons_h(x0) == [[2.0 0.0; 0.0 2.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2, x[2] * sin(x[1]) - x[1]]
    optf = OptimizationFunction{false}(
        rosenbrock,
        OptimizationBase.AutoForwardDiff(),
        cons = cons
    )
    optprob = OptimizationBase.instantiate_function(
        optf, x0,
        OptimizationBase.AutoForwardDiff(),
        nothing, 2, g = true, h = true, cons_j = true, cons_h = true
    )

    @test optprob.grad(x0) == G1
    @test optprob.hess(x0) == H1
    @test optprob.cons(x0) == [0.0, 0.0]
    @test optprob.cons_j([5.0, 3.0]) ≈ [10.0 6.0; -0.149013 -0.958924] rtol = 1.0e-6
    @test optprob.cons_h(x0) == [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2]
    optf = OptimizationFunction{false}(
        rosenbrock,
        OptimizationBase.AutoReverseDiff(),
        cons = cons
    )
    optprob = OptimizationBase.instantiate_function(
        optf, x0,
        OptimizationBase.AutoReverseDiff(),
        nothing, 1, g = true, h = true, cons_j = true, cons_h = true
    )

    @test optprob.grad(x0) == G1
    @test optprob.hess(x0) == H1

    @test optprob.cons(x0) == [0.0]

    @test optprob.cons_j([5.0, 3.0]) == [10.0, 6.0]

    @test optprob.cons_h(x0) == [[2.0 0.0; 0.0 2.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2, x[2] * sin(x[1]) - x[1]]
    optf = OptimizationFunction{false}(
        rosenbrock,
        OptimizationBase.AutoReverseDiff(),
        cons = cons
    )
    optprob = OptimizationBase.instantiate_function(
        optf, x0,
        OptimizationBase.AutoReverseDiff(),
        nothing, 2, g = true, h = true, cons_j = true, cons_h = true
    )

    @test optprob.grad(x0) == G1
    @test optprob.hess(x0) == H1
    @test optprob.cons(x0) == [0.0, 0.0]
    @test optprob.cons_j([5.0, 3.0]) ≈ [10.0 6.0; -0.149013 -0.958924] rtol = 1.0e-6
    @test optprob.cons_h(x0) == [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2]
    optf = OptimizationFunction{false}(
        rosenbrock,
        OptimizationBase.AutoReverseDiff(; compile = true),
        cons = cons
    )
    optprob = OptimizationBase.instantiate_function(
        optf, x0,
        OptimizationBase.AutoReverseDiff(; compile = true),
        nothing, 1, g = true, h = true, cons_j = true, cons_h = true
    )

    @test optprob.grad(x0) == G1
    @test optprob.hess(x0) == H1

    @test optprob.cons(x0) == [0.0]

    @test optprob.cons_j([5.0, 3.0]) == [10.0, 6.0]

    @test optprob.cons_h(x0) == [[2.0 0.0; 0.0 2.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2, x[2] * sin(x[1]) - x[1]]
    optf = OptimizationFunction{false}(
        rosenbrock,
        OptimizationBase.AutoReverseDiff(; compile = true),
        cons = cons
    )
    optprob = OptimizationBase.instantiate_function(
        optf, x0,
        OptimizationBase.AutoReverseDiff(; compile = true),
        nothing, 2, g = true, h = true, cons_j = true, cons_h = true
    )

    @test optprob.grad(x0) == G1
    @test optprob.hess(x0) == H1
    @test optprob.cons(x0) == [0.0, 0.0]
    @test optprob.cons_j([5.0, 3.0]) ≈ [10.0 6.0; -0.149013 -0.958924] rtol = 1.0e-6
    @test optprob.cons_h(x0) == [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2]
    optf = OptimizationFunction{false}(
        rosenbrock,
        AutoSparse(OptimizationBase.AutoForwardDiff()),
        cons = cons
    )
    optprob = OptimizationBase.instantiate_function(
        optf, x0,
        AutoSparse(OptimizationBase.AutoForwardDiff()),
        nothing, 1, g = true, h = true, cons_j = true, cons_h = true
    )

    @test optprob.grad(x0) == G1
    @test Array(optprob.hess(x0)) ≈ H1

    @test optprob.cons(x0) == [0.0]

    @test optprob.cons_j([5.0, 3.0]) == [10.0, 6.0]

    @test optprob.cons_h(x0) == [[2.0 0.0; 0.0 2.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2, x[2] * sin(x[1]) - x[1]]
    optf = OptimizationFunction{false}(
        rosenbrock,
        AutoSparse(OptimizationBase.AutoForwardDiff()),
        cons = cons
    )
    optprob = OptimizationBase.instantiate_function(
        optf, x0,
        AutoSparse(OptimizationBase.AutoForwardDiff()),
        nothing, 2, g = true, h = true, cons_j = true, cons_h = true
    )

    @test optprob.grad(x0) == G1
    @test Array(optprob.hess(x0)) ≈ H1
    @test optprob.cons(x0) == [0.0, 0.0]
    @test Array(optprob.cons_j([5.0, 3.0])) ≈ [10.0 6.0; -0.149013 -0.958924] rtol = 1.0e-6
    @test Array.(optprob.cons_h(x0)) ≈ [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2]
    optf = OptimizationFunction{false}(
        rosenbrock,
        AutoSparse(OptimizationBase.AutoFiniteDiff()),
        cons = cons
    )
    optprob = OptimizationBase.instantiate_function(
        optf, x0,
        AutoSparse(OptimizationBase.AutoFiniteDiff()),
        nothing, 1, g = true, h = true, cons_j = true, cons_h = true
    )

    @test optprob.grad(x0) ≈ G1 rtol = 1.0e-4
    @test Array(optprob.hess(x0)) ≈ H1

    @test optprob.cons(x0) == [0.0]

    @test optprob.cons_j([5.0, 3.0]) ≈ [10.0, 6.0]

    @test optprob.cons_h(x0) == [[2.0 0.0; 0.0 2.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2, x[2] * sin(x[1]) - x[1]]
    optf = OptimizationFunction{false}(
        rosenbrock,
        AutoSparse(OptimizationBase.AutoFiniteDiff()),
        cons = cons
    )
    optprob = OptimizationBase.instantiate_function(
        optf, x0,
        AutoSparse(OptimizationBase.AutoForwardDiff()),
        nothing, 2, g = true, h = true, cons_j = true, cons_h = true
    )

    @test optprob.grad(x0) == G1
    @test Array(optprob.hess(x0)) ≈ H1
    @test optprob.cons(x0) == [0.0, 0.0]
    @test Array(optprob.cons_j([5.0, 3.0])) ≈ [10.0 6.0; -0.149013 -0.958924] rtol = 1.0e-6
    @test Array.(optprob.cons_h(x0)) ≈ [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2]
    optf = OptimizationFunction{false}(
        rosenbrock,
        AutoSparse(OptimizationBase.AutoReverseDiff()),
        cons = cons
    )
    optprob = OptimizationBase.instantiate_function(
        optf, x0,
        AutoSparse(OptimizationBase.AutoReverseDiff()),
        nothing, 1, g = true, h = true, cons_j = true, cons_h = true
    )

    @test optprob.grad(x0) == G1
    @test optprob.hess(x0) == H1

    @test optprob.cons(x0) == [0.0]

    @test optprob.cons_j([5.0, 3.0]) == [10.0, 6.0]

    @test optprob.cons_h(x0) == [[2.0 0.0; 0.0 2.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2, x[2] * sin(x[1]) - x[1]]
    optf = OptimizationFunction{false}(
        rosenbrock,
        AutoSparse(OptimizationBase.AutoReverseDiff()),
        cons = cons
    )
    optprob = OptimizationBase.instantiate_function(
        optf, x0,
        AutoSparse(OptimizationBase.AutoReverseDiff()),
        nothing, 2, g = true, h = true, cons_j = true, cons_h = true
    )

    @test optprob.grad(x0) == G1
    @test Array(optprob.hess(x0)) ≈ H1
    @test optprob.cons(x0) == [0.0, 0.0]
    @test Array(optprob.cons_j([5.0, 3.0])) ≈ [10.0 6.0; -0.149013 -0.958924] rtol = 1.0e-6
    @test Array.(optprob.cons_h(x0)) ≈ [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2]
    optf = OptimizationFunction{false}(
        rosenbrock,
        AutoSparse(OptimizationBase.AutoReverseDiff(; compile = true)),
        cons = cons
    )
    optprob = OptimizationBase.instantiate_function(
        optf, x0,
        AutoSparse(OptimizationBase.AutoReverseDiff(; compile = true)),
        nothing, 1, g = true, h = true, cons_j = true, cons_h = true
    )

    @test optprob.grad(x0) == G1
    @test optprob.hess(x0) == H1
    @test optprob.cons(x0) == [0.0]

    @test optprob.cons_j([5.0, 3.0]) == [10.0, 6.0]

    @test optprob.cons_h(x0) == [[2.0 0.0; 0.0 2.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2, x[2] * sin(x[1]) - x[1]]
    optf = OptimizationFunction{false}(
        rosenbrock,
        AutoSparse(OptimizationBase.AutoReverseDiff(; compile = true)),
        cons = cons
    )
    optprob = OptimizationBase.instantiate_function(
        optf, x0,
        AutoSparse(OptimizationBase.AutoReverseDiff(; compile = true)),
        nothing, 2, g = true, h = true, cons_j = true, cons_h = true
    )

    @test optprob.grad(x0) == G1
    @test Array(optprob.hess(x0)) ≈ H1
    @test optprob.cons(x0) == [0.0, 0.0]
    @test Array(optprob.cons_j([5.0, 3.0])) ≈ [10.0 6.0; -0.149013 -0.958924] rtol = 1.0e-6
    @test Array.(optprob.cons_h(x0)) ≈ [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2]
    optf = OptimizationFunction{false}(
        rosenbrock,
        AutoZygote(),
        cons = cons
    )
    optprob = OptimizationBase.instantiate_function(
        optf, x0, AutoZygote(),
        nothing, 1, g = true, h = true, cons_j = true, cons_h = true
    )

    @test optprob.grad(x0) == G1
    @test optprob.hess(x0) == H1
    @test optprob.cons(x0) == [0.0]

    @test optprob.cons_j([5.0, 3.0]) == [10.0, 6.0]

    @test optprob.cons_h(x0) == [[2.0 0.0; 0.0 2.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2, x[2] * sin(x[1]) - x[1]]
    optf = OptimizationFunction{false}(
        rosenbrock,
        AutoZygote(),
        cons = cons
    )
    optprob = OptimizationBase.instantiate_function(
        optf, x0, AutoZygote(),
        nothing, 2, g = true, h = true, cons_j = true, cons_h = true
    )

    @test optprob.grad(x0) == G1
    @test Array(optprob.hess(x0)) ≈ H1
    @test optprob.cons(x0) == [0.0, 0.0]
    @test optprob.cons_j([5.0, 3.0]) ≈ [10.0 6.0; -0.149013 -0.958924] rtol = 1.0e-6
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
        optf, rand(3), AutoForwardDiff(), iterate(data)[1], g = true, fg = true
    )
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
        @test G ≈ G1 rtol = 1.0e-6
    end
    @test G0 ≈ sum(stochgrads) rtol = 1.0e-1

    optf = OptimizationFunction(loss, AutoReverseDiff())
    optf = OptimizationBase.instantiate_function(
        optf, rand(3), AutoReverseDiff(), iterate(data)[1], g = true, fg = true
    )
    G0 = zeros(3)
    optf.grad(G0, ones(3), (x0, y0))
    stochgrads = []
    for (x, y) in data
        G = zeros(3)
        optf.grad(G, ones(3), (x, y))
        push!(stochgrads, copy(G))
        G1 = zeros(3)
        optf.fg(G1, ones(3), (x, y))
        @test G ≈ G1 rtol = 1.0e-6
    end
    @test G0 ≈ sum(stochgrads) rtol = 1.0e-1

    optf = OptimizationFunction(loss, AutoZygote())
    optf = OptimizationBase.instantiate_function(
        optf, rand(3), AutoZygote(), iterate(data)[1], g = true, fg = true
    )
    G0 = zeros(3)
    optf.grad(G0, ones(3), (x0, y0))
    stochgrads = []
    for (x, y) in data
        G = zeros(3)
        optf.grad(G, ones(3), (x, y))
        push!(stochgrads, copy(G))
        G1 = zeros(3)
        optf.fg(G1, ones(3), (x, y))
        @test G ≈ G1 rtol = 1.0e-6
    end
    @test G0 ≈ sum(stochgrads) rtol = 1.0e-1

    optf = OptimizationFunction(loss, AutoEnzyme())
    optf = OptimizationBase.instantiate_function(
        optf, rand(3), AutoEnzyme(mode = set_runtime_activity(Reverse)),
        iterate(data)[1], g = true, fg = true
    )
    G0 = zeros(3)
    optf.grad(G0, ones(3), (x0, y0))
    stochgrads = []
    for (x, y) in data
        G = zeros(3)
        optf.grad(G, ones(3), (x, y))
        push!(stochgrads, copy(G))
        G1 = zeros(3)
        optf.fg(G1, ones(3), (x, y))
        @test G ≈ G1 rtol = 1.0e-6
    end
    @test G0 ≈ sum(stochgrads) rtol = 1.0e-1
end

@testset "user-supplied grad is used by the generated fg!" begin
    # A supplied `f.grad` must drive `fg!` rather than being replaced by a fresh AD
    # preparation: rebuilding it silently discards whatever tuning the caller did
    # (SciML/Optimization.jl#1282).
    ncalls = Ref(0)
    rosen(x, p = nothing) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
    function rosen_grad!(res, x, p = nothing)
        ncalls[] += 1
        res[1] = -2 * (1 - x[1]) - 400 * x[1] * (x[2] - x[1]^2)
        res[2] = 200 * (x[2] - x[1]^2)
        return res
    end

    Gref = zeros(2)
    rosen_grad!(Gref, [0.5, 0.7])

    adtypes = (
        AutoForwardDiff(), AutoReverseDiff(), AutoZygote(), AutoEnzyme(),
        AutoSparse(AutoZygote()),
    )
    for adtype in adtypes, g in (false, true)
        optf = OptimizationBase.instantiate_function(
            OptimizationFunction(rosen, adtype; grad = rosen_grad!),
            zeros(2), adtype, nothing; g = g, fg = true
        )
        optf.fg === nothing && continue
        ncalls[] = 0
        G = zeros(2)
        y = optf.fg(G, [0.5, 0.7])
        @test ncalls[] == 1
        @test y ≈ rosen([0.5, 0.7])
        @test G ≈ Gref

        # Without a user gradient the AD path still has to build its own preparation,
        # including when `fg` is requested but `g` is not.
        optf_ad = OptimizationBase.instantiate_function(
            OptimizationFunction(rosen, adtype), zeros(2), adtype, nothing; g = g, fg = true
        )
        Gad = zeros(2)
        @test optf_ad.fg(Gad, [0.5, 0.7]) ≈ rosen([0.5, 0.7])
        @test Gad ≈ Gref
    end
end

@testset "Enzyme out-of-place grad/fg wiring" begin
    rosen(x, p = nothing) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
    rosen_grad(x, p = nothing) = [
        -2 * (1 - x[1]) - 400 * x[1] * (x[2] - x[1]^2),
        200 * (x[2] - x[1]^2),
    ]
    ad = AutoEnzyme()
    z = [0.5, 0.7]
    gref = rosen_grad(z)

    # `g` alone must honour a supplied gradient; it used to be gated on `fg`.
    optf = OptimizationBase.instantiate_function(
        OptimizationFunction{false}(rosen, ad; grad = rosen_grad),
        zeros(2), ad, nothing; g = true, fg = false
    )
    @test optf.grad !== nothing
    @test optf.grad(z) ≈ gref

    # The AD `fg!` must return the buffer it differentiated into, with or without `g`.
    for g in (false, true)
        optf = OptimizationBase.instantiate_function(
            OptimizationFunction{false}(rosen, ad), zeros(2), ad, nothing; g = g, fg = true
        )
        y, G = optf.fg(z)
        @test y ≈ rosen(z)
        @test G ≈ gref
    end
end

@testset "Enzyme Hessian batch cap" begin
    n = 17
    x = collect(range(0.1, 1.7; length = n))
    quartic(x, p = nothing) = sum(abs2(abs2(xi)) for xi in x)
    expected_hessian = Matrix(Diagonal(12 .* x .^ 2))

    enzyme_ext = Base.get_extension(OptimizationBase, :OptimizationEnzymeExt)
    @test enzyme_ext._hessian_batch_width(4) == 4
    @test enzyme_ext._hessian_batch_width(n) == 8

    θ = ComponentVector(x = x)
    optf = OptimizationBase.instantiate_function(
        OptimizationFunction(quartic, AutoEnzyme()), θ, AutoEnzyme(), nothing;
        h = true, fgh = true
    )
    gradient = similar(θ)
    hessian = similar(expected_hessian)
    optf.hess(hessian, θ)
    @test hessian ≈ expected_hessian
    optf.fgh(gradient, hessian, θ)
    @test hessian ≈ expected_hessian

    optf_oop = OptimizationBase.instantiate_function(
        OptimizationFunction{false}(quartic, AutoEnzyme()), x, AutoEnzyme(), nothing;
        h = true, fgh = true
    )
    @test optf_oop.hess(x) ≈ expected_hessian
    _, hessian_oop = optf_oop.fgh(x)
    @test hessian_oop ≈ expected_hessian

    function quadratic_constraints!(res, x, p)
        first_constraint = zero(eltype(x))
        second_constraint = zero(eltype(x))
        for i in eachindex(x)
            first_constraint += x[i]^2
            second_constraint += i * x[i]^2
        end
        res[1] = first_constraint
        res[2] = second_constraint
        return
    end
    σ = 0.7
    μ = [0.2, -0.1]
    expected_lagrangian_hessian =
        σ .* expected_hessian .+ Matrix(Diagonal(2μ[1] .+ 2μ[2] .* eachindex(x)))
    optf_lagrangian = OptimizationBase.instantiate_function(
        OptimizationFunction(quartic, AutoEnzyme(); cons = quadratic_constraints!),
        x, AutoEnzyme(), nothing, 2; lag_h = true
    )
    lagrangian_hessian = similar(expected_hessian)
    optf_lagrangian.lag_h(lagrangian_hessian, x, σ, μ)
    @test lagrangian_hessian ≈ expected_lagrangian_hessian
    packed_lagrangian_hessian = Vector{eltype(x)}(undef, n * (n + 1) ÷ 2)
    optf_lagrangian.lag_h(packed_lagrangian_hessian, x, σ, μ)
    @test packed_lagrangian_hessian ≈ vcat(
        [expected_lagrangian_hessian[i, 1:i] for i in 1:n]...
    )
    @test any(
        getfield(optf_lagrangian.lag_h, i) isa Val{8}
            for i in 1:fieldcount(typeof(optf_lagrangian.lag_h))
    )

end
