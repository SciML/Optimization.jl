include(joinpath(@__DIR__, "..", "despecialize_common.jl"))
using ChainRulesCore, Enzyme

# Not a singleton, so `Duplicated` is a valid annotation for it (Enzyme rejects
# `Duplicated` of a ghost type like `typeof(rosenbrock_cons!)`)
struct ScaledCons
    scale::Vector{Float64}
end
function (c::ScaledCons)(res, x, p)
    res[1] = c.scale[1] * (x[1]^2 + x[2]^2)
    res[2] = p.a * x[1] * x[2]
    return nothing
end

@testset "despecialize: Enzyme" begin
    @test Base.get_extension(OptimizationBase, :OptimizationEnzymeExt) !== nothing

    @testset "derivatives" begin
        # `inner_cons` only accepts `NullParameters`/`nothing` for `p`
        test_derivatives(AutoEnzyme(); broken = (:cons_h,))
    end

    @testset "closure types do not depend on the problem" begin
        for (flags, field) in (
                ((; g = true), :grad), ((; fg = true), :fg), ((; h = true), :hess),
                ((; hv = true), :hv), ((; cons_j = true), :cons_j),
                ((; cons_jvp = true), :cons_jvp), ((; cons_vjp = true), :cons_vjp),
                ((; cons_h = true), :cons_h), ((; lag_h = true), :lag_h),
            )
            a = despecialized_instance(AutoDespecialize, AutoEnzyme(); flags...)
            b = despecialized_instance(
                AutoDespecialize, AutoEnzyme(), shifted_rosenbrock, other_cons!,
                OtherParams(1.0, 100.0, 1); flags...
            )
            @test typeof(getfield(a, field)) == typeof(getfield(b, field))
            @test typeof(a) == typeof(b)
        end
    end

    @testset "constraint Jacobian with function_annotation = $annotation" for annotation in (
            nothing, Enzyme.Const, Enzyme.Duplicated, Enzyme.DuplicatedNoNeed,
        )
        ad = annotation === nothing ? AutoEnzyme() : AutoEnzyme(; function_annotation = annotation)
        cons = ScaledCons([2.0])
        ref = ForwardDiff.jacobian(y -> (r = zeros(eltype(y), 2); cons(r, y, P0); r), X0)
        for spec in (FullSpecialize, AutoDespecialize)
            fi = despecialized_instance(spec, ad, rosenbrock, cons, P0; cons_j = true)
            for _ in 1:2
                J = zeros(2, 2)
                fi.cons_j(J, X0)
                @test J ≈ ref
            end
        end
    end
end
