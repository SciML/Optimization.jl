include(joinpath(@__DIR__, "..", "despecialize_common.jl"))
using ChainRulesCore, Enzyme, ModelingToolkit, Symbolics, Zygote
using ADTypes: AutoSparse, AutoSymbolics
using DifferentiationInterface: SecondOrder

# Problems whose AD choice needs the concrete functions keep their full specialization
# and give the same derivatives as with `FullSpecialize`.
@testset "despecialize: fallbacks" begin
    vector_rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    function vector_cons!(res, x, p)
        res[1] = x[1]^2 + x[2]^2
        res[2] = p[1] * x[1] * x[2]
        return nothing
    end
    ref = reference_derivatives()
    cases = (
        ("Symbolics", AutoSymbolics(), vector_rosenbrock, vector_cons!, [1.0, 100.0]),
        (
            "Sparse(Symbolics)", AutoSparse(AutoSymbolics()), vector_rosenbrock,
            vector_cons!, [1.0, 100.0],
        ),
        ("Sparse(Enzyme)", AutoSparse(AutoEnzyme()), rosenbrock, rosenbrock_cons!, P0),
        (
            "SecondOrder(ForwardDiff, Enzyme)", SecondOrder(AutoForwardDiff(), AutoEnzyme()),
            rosenbrock, rosenbrock_cons!, P0,
        ),
    )
    @testset "$name" for (name, ad, obj, cons, p) in cases
        @test !OptimizationBase.despecializes(OptimizationFunction{true, AutoDespecialize}(obj, ad))
        for op in (:grad, :fg, :cons_j)
            flags, evaluate = DERIVATIVES[op]
            full = evaluate(despecialized_instance(FullSpecialize, ad, obj, cons, p; flags...))
            auto = evaluate(despecialized_instance(AutoDespecialize, ad, obj, cons, p; flags...))
            @test approx(full, ref[op]; rtol = 1.0e-8)
            @test approx(auto, ref[op]; rtol = 1.0e-8)
        end
    end
end
