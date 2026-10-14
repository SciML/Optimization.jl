using OptimizationBase, Test, SciMLBase, ForwardDiff
using OptimizationBase: OptimizationCache, reinit!

# Regression tests for #1383: instantiated derivative/constraint closures must use the
# live `p` after `reinit!`, and a data iterator must never silently fit its first batch.

struct LiveParamsAlg end
SciMLBase.requiresgradient(::LiveParamsAlg) = true
SciMLBase.allowsfg(::LiveParamsAlg) = true
SciMLBase.allowsconstraints(::LiveParamsAlg) = true
SciMLBase.requiresconsjac(::LiveParamsAlg) = true

objective(x, p) = (p[1] - x[1])^2
grad!(G, x, p) = (G[1] = -2 * (p[1] - x[1]); nothing)
cons!(res, x, p) = (res[1] = p[1] * x[1]; nothing)

function live_cache(optf, p; kwargs...)
    prob = OptimizationProblem(optf, [0.0], p; kwargs...)
    return OptimizationCache(prob, LiveParamsAlg())
end

@testset "live p after reinit!" begin
    for optf in (
            OptimizationFunction(objective, OptimizationBase.AutoForwardDiff(); cons = cons!),
            OptimizationFunction(objective; grad = grad!),
        )
        cache = optf.cons === nothing ? live_cache(optf, [1.0]) :
            live_cache(optf, [1.0]; lcons = [-Inf], ucons = [Inf])
        G = zeros(1)

        cache.f.grad(G, [0.0])
        @test G ≈ [-2.0]
        cache.f.grad(G, [0.0], cache.p)
        @test G ≈ [-2.0]

        reinit!(cache; p = [2.0])
        cache.f.grad(G, [0.0])                  # call form without `p`
        @test G ≈ [-4.0]
        cache.f.grad(G, [0.0], cache.p)         # call form with the live `p`
        @test G ≈ [-4.0]
        cache.f.grad(G, [0.0], [5.0])           # an explicit, different `p` is forwarded
        @test G ≈ [-10.0]
        if cache.f.fg !== nothing
            @test cache.f.fg(G, [0.0]) ≈ 4.0
            @test G ≈ [-4.0]
        end
        if cache.f.cons_j !== nothing
            J = zeros(1, 1)
            cache.f.cons_j(J, [1.0])
            @test J ≈ [2.0;;]
        end
    end
end

@testset "re-instantiation only when p is replaced" begin
    optf = OptimizationFunction(objective, OptimizationBase.AutoForwardDiff())
    p = [1.0]
    cache = live_cache(optf, p)
    live = cache.f.grad.live
    G = zeros(1)

    cache.f.grad(G, [0.0])
    f1 = live.f
    cache.f.grad(G, [0.0])
    @test live.f === f1                         # steady state: no rebuild

    cache.p[1] = 3.0                            # in-place mutation is already visible
    cache.f.grad(G, [0.0])
    @test G ≈ [-6.0]
    @test live.f === f1

    reinit!(cache; p = [2.0])
    cache.f.grad(G, [0.0])
    @test G ≈ [-4.0]
    @test live.f !== f1                         # rebuilt once
    f2 = live.f
    cache.f.grad(G, [0.0])
    @test live.f === f2
end

struct LiveHessAlg end
SciMLBase.requiresgradient(::LiveHessAlg) = true
SciMLBase.requireshessian(::LiveHessAlg) = true

@testset "reinit! that changes the sparsity structure" begin
    obj(x, p) = p[1] > 0 ? x[1]^2 : x[2]^2
    optf = OptimizationFunction(obj, AutoSparse(OptimizationBase.AutoForwardDiff()))
    cache = OptimizationCache(OptimizationProblem(optf, [1.0, 1.0], [1.0]), LiveHessAlg())
    H = similar(cache.f.hess_prototype, Float64)
    cache.f.hess(H, [1.0, 1.0])
    @test Matrix(H) ≈ [2.0 0.0; 0.0 0.0]

    reinit!(cache; p = [2.0])                   # same structure: rebuilt and used
    cache.f.hess(H, [1.0, 1.0])
    @test Matrix(H) ≈ [2.0 0.0; 0.0 0.0]

    reinit!(cache; p = [-1.0])                  # structure differs from the declared one
    @test_throws ArgumentError cache.f.hess(H, [1.0, 1.0])
end

@testset "NoAD closure signatures" begin
    vjp!(Jv, x, v, p) = (Jv .= p[1] .* v; nothing)
    lagh!(H, x, σ, μ, p) = (H .= σ * p[1] + μ[1]; nothing)
    optf = OptimizationFunction(
        objective; grad = grad!, cons = cons!, cons_vjp = vjp!, cons_jvp = vjp!,
        lag_h = lagh!
    )
    fi = OptimizationBase.instantiate_function(
        optf, [0.0], SciMLBase.NoAD(), [2.0], 1
    )
    Jv = zeros(1)
    fi.cons_vjp(Jv, [0.0], [3.0])
    @test Jv ≈ [6.0]
    fi.cons_jvp(Jv, [0.0], [3.0], [5.0])
    @test Jv ≈ [15.0]
    H = zeros(1, 1)
    fi.lag_h(H, [0.0], 1.0, [1.0])              # `lag_h(res, θ, σ, μ[, p])`
    @test H ≈ [3.0;;]
    fi.lag_h(H, [0.0], 1.0, [1.0], [4.0])
    @test H ≈ [5.0;;]

    # Without parameters, the call form with `p` works as well.
    fi0 = OptimizationBase.instantiate_function(
        OptimizationFunction((x, p) -> x[1]^2; grad = (G, x, p) -> (G[1] = 2x[1])),
        [1.0], SciMLBase.NoAD(), SciMLBase.NullParameters()
    )
    G = zeros(1)
    fi0.grad(G, [1.0])
    @test G ≈ [2.0]
    fi0.grad(G, [3.0], SciMLBase.NullParameters())
    @test G ≈ [6.0]
end

# A minimal data iterator over batches of the scalar data points.
struct TestBatches
    batches::Vector{Vector{Float64}}
end
Base.iterate(b::TestBatches, i = 1) = i > length(b.batches) ? nothing : (b.batches[i], i + 1)
Base.length(b::TestBatches) = length(b.batches)
OptimizationBase.isa_dataiterator(::TestBatches) = true

@testset "data iterators" begin
    data = TestBatches([[1.0, 1.0], [5.0, 5.0]])
    loss(u, batch) = sum(abs2, u[1] .- batch)
    for adtype in (OptimizationBase.AutoForwardDiff(), SciMLBase.NoAD())
        optf = adtype isa SciMLBase.NoAD ?
            OptimizationFunction(loss; grad = (G, u, b) -> (G[1] = sum(2 .* (u[1] .- b)))) :
            OptimizationFunction(loss, adtype)
        cache = live_cache(optf, data)
        @test cache.p === data
        G = zeros(1)

        # A solver that does not select a batch must not silently use the first one.
        @test_throws ArgumentError cache.f.grad(G, [0.0])
        @test_throws ArgumentError cache.f.grad(G, [0.0], data)

        # Minibatch-aware solvers pass the batch explicitly.
        cache.f.grad(G, [0.0], [5.0, 5.0])
        @test G ≈ [-20.0]
    end
end
