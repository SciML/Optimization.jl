include("despecialize_common.jl")
using FiniteDiff, ReverseDiff, Zygote

const DespecializedDataIterator = OptimizationBase.DespecializedDataIterator

struct BatchIterator
    batches::Vector{RosenbrockParams}
end
OptimizationBase.isa_dataiterator(::BatchIterator) = true
Base.iterate(it::BatchIterator, state...) = iterate(it.batches, state...)
Base.length(it::BatchIterator) = length(it.batches)

# A solver that needs gradients, to build an `OptimizationCache` without a solver package
struct GradientSolver end
SciMLBase.requiresgradient(::GradientSolver) = true
SciMLBase.allowsfg(::GradientSolver) = true

@testset "despecialize" begin
    grad!(G, x, p) = (G .= ForwardDiff.gradient(y -> rosenbrock(y, p), x); nothing)
    @testset "wrapping" begin
        for spec in (FullSpecialize, AutoSpecialize, AutoDespecialize)
            of = OptimizationFunction{true, spec}(
                rosenbrock, AutoForwardDiff(); grad = grad!, cons = rosenbrock_cons!
            )
            f, p = OptimizationBase.despecialize(of, P0)
            erased = spec !== FullSpecialize
            @test OptimizationBase.despecializes(of) == erased
            @test (f.f isa DespecializedFunction{:objective}) == erased
            @test (f.grad isa DespecializedFunction{:inplace}) == erased
            @test (f.cons isa DespecializedFunction{:inplace}) == erased
            @test f.hess === nothing
            @test SciMLBase.specialization(f) === spec
            @test (p isa DespecializedParameters) == (spec === AutoDespecialize)
            @test SciMLBase.unwrap_parameters(p) === P0
            @test SciMLBase.unwrapped_f(f.f) === rosenbrock
            # calling the wrappers unwraps `p`
            @test f.f(X0, p) == rosenbrock(X0, P0)
            G = zeros(2)
            f.grad(G, X0, p)
            @test G ≈ reference_derivatives().grad
            # despecializing again changes nothing
            f2, p2 = OptimizationBase.despecialize(f, p)
            @test typeof(f2) == typeof(f)
            @test p2 === p
        end
        # out-of-place functions keep their specialization
        oop = OptimizationFunction{false, AutoDespecialize}(rosenbrock, AutoForwardDiff())
        @test !OptimizationBase.despecializes(oop)
        @test OptimizationBase.despecialize(oop, P0) === (oop, P0)
        # nothing to hide in `NullParameters`
        f, p = OptimizationBase.despecialize(
            OptimizationFunction{true, AutoDespecialize}((x, p) -> sum(abs2, x)),
            SciMLBase.NullParameters()
        )
        @test p === SciMLBase.NullParameters()
    end

    @testset "data iterators" begin
        data = BatchIterator([RosenbrockParams(1.0, 100.0), RosenbrockParams(2.0, 10.0)])
        of = OptimizationFunction{true, AutoDespecialize}(rosenbrock, AutoForwardDiff())
        f, p = OptimizationBase.despecialize(of, data)
        @test p isa DespecializedDataIterator
        @test OptimizationBase.isa_dataiterator(p)
        @test length(p) == 2
        @test SciMLBase.unwrap_parameters(p) === data
        batches = collect(p)
        @test all(b -> b isa DespecializedParameters, batches)
        @test SciMLBase.unwrap_parameters.(batches) == data.batches
    end

    @testset "value type" begin
        f, p = OptimizationBase.despecialize(
            OptimizationFunction{true, AutoDespecialize}((x, p) -> Float64(sum(x))), P0
        )
        @test_throws ErrorException f.f(Float32[1, 2], p)
        @test f.f([1.0, 2.0], p) === 3.0
    end

    @testset "cache types do not depend on the problem" begin
        caches = map(
            (
                (rosenbrock, rosenbrock_cons!, P0),
                (shifted_rosenbrock, other_cons!, OtherParams(1.0, 100.0, 1)),
            )
        ) do (obj, cons, p)
            of = OptimizationFunction{true, AutoDespecialize}(obj, AutoForwardDiff(); cons)
            prob = OptimizationProblem(of, copy(X0), p; lcons = [-Inf, -Inf], ucons = [Inf, Inf])
            OptimizationBase.OptimizationCache(prob, GradientSolver())
        end
        @test typeof(caches[1]) == typeof(caches[2])
        @test caches[1].f.f isa DespecializedFunction
        @test caches[1].p isa DespecializedParameters
        G = zeros(2)
        caches[1].f.grad(G, X0)
        @test G ≈ reference_derivatives().grad
    end

    @testset "reinit!" begin
        of = OptimizationFunction{true, AutoDespecialize}(rosenbrock, AutoForwardDiff())
        cache = OptimizationBase.OptimizationCache(
            OptimizationProblem(of, copy(X0), P0), GradientSolver()
        )
        # a new parameter object, also of another type
        for newp in (RosenbrockParams(2.0, 10.0), OtherParams(3.0, 1.0, 0))
            SciMLBase.reinit!(cache; p = newp)
            @test cache.p isa DespecializedParameters
            @test SciMLBase.unwrap_parameters(cache.p) === newp
            @test cache.f.f(X0, cache.p) == rosenbrock(X0, newp)
        end
        data = BatchIterator([P0])
        cache = OptimizationBase.OptimizationCache(
            OptimizationProblem(of, copy(X0), data), GradientSolver()
        )
        newdata = BatchIterator([RosenbrockParams(2.0, 10.0)])
        SciMLBase.reinit!(cache; p = newdata)
        @test cache.p isa DespecializedDataIterator
        @test SciMLBase.unwrap_parameters(cache.p) === newdata
    end

    @testset "derivatives: $name" for (name, ad, kw) in (
            ("ForwardDiff", AutoForwardDiff(), (;)),
            ("ReverseDiff", AutoReverseDiff(), (;)),
            ("ReverseDiff (compiled tape)", AutoReverseDiff(; compile = true), (;)),
            ("Zygote", AutoZygote(), (;)),
            ("FiniteDiff", AutoFiniteDiff(), (; ops = (:grad, :fg, :cons_j, :cons_jvp, :cons_vjp), rtol = 1.0e-4)),
        )
        test_derivatives(ad; kw...)
    end
end
