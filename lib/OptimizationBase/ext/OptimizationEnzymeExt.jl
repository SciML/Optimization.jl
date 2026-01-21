module OptimizationEnzymeExt

import OptimizationBase, OptimizationBase.ArrayInterface
import SciMLBase: OptimizationFunction
import SciMLBase
import OptimizationBase.LinearAlgebra: I, dot
import OptimizationBase.ADTypes: AutoEnzyme
using Enzyme
using Core: Vararg

@inline function firstapply(f::F, θ, p) where {F}
    res = f(θ, p)
    return if isa(res, AbstractFloat)
        res
    else
        first(res)
    end
end

function inner_grad(mode::Mode, θ, bθ, f, p) where {Mode}
    Enzyme.autodiff(
        mode,
        Const(firstapply),
        Active,
        Const(f),
        Enzyme.Duplicated(θ, bθ),
        Const(p)
    )
    return nothing
end

function hv_f2_alloc(mode::Mode, xdup, f, p) where {Mode}
    Enzyme.autodiff(
        mode,
        Const(firstapply),
        Active,
        Const(f),
        xdup,
        Const(p)
    )
    return xdup
end

function inner_cons(
        x, fcons::Function, p::Union{SciMLBase.NullParameters, Nothing},
        num_cons::Int, i::Int
    )
    res = zeros(eltype(x), num_cons)
    fcons(res, x, p)
    return res[i]
end

function cons_f2(mode, x, dx, fcons, p, num_cons, i)
    Enzyme.autodiff_deferred(
        mode, Const(inner_cons), Active, Enzyme.Duplicated(x, dx),
        Const(fcons), Const(p), Const(num_cons), Const(i)
    )
    return nothing
end

function inner_cons_oop(
        x::Vector{T}, fcons::Function, p::Union{SciMLBase.NullParameters, Nothing},
        i::Int
    ) where {T}
    return fcons(x, p)[i]
end

function cons_f2_oop(mode, x, dx, fcons, p, i)
    Enzyme.autodiff_deferred(
        mode, Const(inner_cons_oop), Active, Enzyme.Duplicated(x, dx),
        Const(fcons), Const(p), Const(i)
    )
    return nothing
end

function lagrangian(x, _f::Function, cons::Function, p, λ, σ = one(eltype(x)))
    res = zeros(eltype(x), length(λ))
    cons(res, x, p)
    return σ * _f(x, p) + dot(λ, res)
end

function lag_grad(mode, x, dx, lagrangian::Function, _f::Function, cons::Function, p, σ, λ)
    Enzyme.autodiff_deferred(
        mode, Const(lagrangian), Active, Enzyme.Duplicated(x, dx),
        Const(_f), Const(cons), Const(p), Const(λ), Const(σ)
    )
    return nothing
end

function set_runtime_activity2(
        a::Mode1, ::Enzyme.Mode{ABI, Err, RTA}
    ) where {Mode1, ABI, Err, RTA}
    return Enzyme.set_runtime_activity(a, RTA)
end
function_annotation(::Nothing) = Nothing
function_annotation(::AutoEnzyme{<:Any, A}) where {A} = A
function OptimizationBase.instantiate_function(
        f::OptimizationFunction{true}, x,
        adtype::AutoEnzyme, p, num_cons = 0;
        g = false, h = false, hv = false, fg = false, fgh = false,
        cons_j = false, cons_vjp = false, cons_jvp = false, cons_h = false,
        lag_h = false
    )
    rmode = if adtype.mode isa Nothing
        Enzyme.Reverse
    else
        set_runtime_activity2(Enzyme.Reverse, adtype.mode)
    end

    fmode = if adtype.mode isa Nothing
        Enzyme.Forward
    else
        set_runtime_activity2(Enzyme.Forward, adtype.mode)
    end

    func_annot = function_annotation(adtype)

    if g == true && f.grad === nothing
        function grad(res, θ, p = p)
            Enzyme.make_zero!(res)
            return Enzyme.autodiff(
                rmode,
                Const(firstapply),
                Active,
                Const(f.f),
                Enzyme.Duplicated(θ, res),
                Const(p)
            )
        end
    elseif g == true
        grad = (G, θ, p = p) -> f.grad(G, θ, p)
    else
        grad = nothing
    end

    if fg == true && f.fg === nothing
        function fg!(res, θ, p = p)
            Enzyme.make_zero!(res)
            y = Enzyme.autodiff(
                WithPrimal(rmode),
                Const(firstapply),
                Active,
                Const(f.f),
                Enzyme.Duplicated(θ, res),
                Const(p)
            )[2]
            return y
        end
    elseif fg == true
        fg! = (res, θ, p = p) -> f.fg(res, θ, p)
    else
        fg! = nothing
    end

    if h == true && f.hess === nothing
        vdθ = Tuple((Array(r) for r in eachrow(I(length(x)) * one(eltype(x)))))
        bθ = zeros(eltype(x), length(x))

        if f.hess_prototype === nothing
            vdbθ = Tuple(zeros(eltype(x), length(x)) for i in eachindex(x))
        else
            #useless right now, looks like there is no way to tell Enzyme the sparsity pattern?
            vdbθ = Tuple((copy(r) for r in eachrow(f.hess_prototype)))
        end

        function hess(res, θ, p = p)
            Enzyme.make_zero!(bθ)
            Enzyme.make_zero!.(vdbθ)

            Enzyme.autodiff(
                fmode,
                inner_grad,
                Const(rmode),
                Enzyme.BatchDuplicated(θ, vdθ),
                Enzyme.BatchDuplicatedNoNeed(bθ, vdbθ),
                Const(f.f),
                Const(p)
            )

            for i in eachindex(θ)
                res[i, :] .= vdbθ[i]
            end
            return
        end
    elseif h == true
        hess = (H, θ, p = p) -> f.hess(H, θ, p)
    else
        hess = nothing
    end

    if fgh == true && f.fgh === nothing
        function fgh!(G, H, θ, p = p)
            vdθ = Tuple((Array(r) for r in eachrow(I(length(θ)) * one(eltype(θ)))))
            vdbθ = Tuple(zeros(eltype(θ), length(θ)) for i in eachindex(θ))

            Enzyme.autodiff(
                fmode,
                inner_grad,
                Const(rmode),
                Enzyme.BatchDuplicated(θ, vdθ),
                Enzyme.BatchDuplicatedNoNeed(G, vdbθ),
                Const(f.f),
                Const(p)
            )

            for i in eachindex(θ)
                H[i, :] .= vdbθ[i]
            end
            return
        end
    elseif fgh == true
        fgh! = (G, H, θ, p = p) -> f.fgh(G, H, θ, p)
    else
        fgh! = nothing
    end

    if hv == true && f.hv === nothing
        function hv!(H, θ, v, p = p)
            dθ = zero(θ)
            Enzyme.make_zero!(H)
            return Enzyme.autodiff(
                fmode,
                inner_grad,
                Const(rmode),
                Duplicated(θ, v),
                Duplicated(dθ, H),
                Const(f.f),
                Const(p)
            )
        end
    elseif hv == true
        hv! = (H, θ, v, p = p) -> f.hv(H, θ, v, p)
    else
        hv! = nothing
    end

    if f.cons === nothing
        cons = nothing
    else
        cons = (res, θ) -> f.cons(res, θ, p)
    end

    if cons !== nothing && cons_j == true && f.cons_j === nothing
        # if num_cons > length(x)
        seeds = Enzyme.onehot(x)
        Jaccache = Tuple(zeros(eltype(x), num_cons) for i in 1:length(x))
        basefunc = f.cons
        if func_annot <: Enzyme.Const
            basefunc = Enzyme.Const(basefunc)
        elseif func_annot <: Enzyme.Duplicated || func_annot <: Enzyme.BatchDuplicated
            basefunc = Enzyme.BatchDuplicated(
                basefunc, Tuple(
                    make_zero(basefunc)
                        for i in 1:length(x)
                )
            )
        elseif func_annot <: Enzyme.DuplicatedNoNeed ||
                func_annot <: Enzyme.BatchDuplicatedNoNeed
            basefunc = Enzyme.BatchDuplicatedNoNeed(
                basefunc, Tuple(
                    make_zero(basefunc)
                        for i in 1:length(x)
                )
            )
        end
        # else
        #     seeds = Enzyme.onehot(zeros(eltype(x), num_cons))
        #     Jaccache = Tuple(zero(x) for i in 1:num_cons)
        # end

        y = zeros(eltype(x), num_cons)

        function cons_j!(J, θ)
            for jc in Jaccache
                Enzyme.make_zero!(jc)
            end
            Enzyme.make_zero!(y)
            if func_annot <: Enzyme.Duplicated || func_annot <: Enzyme.BatchDuplicated ||
                    func_annot <: Enzyme.DuplicatedNoNeed ||
                    func_annot <: Enzyme.BatchDuplicatedNoNeed
                for bf in basefunc.dval
                    Enzyme.make_zero!(bf)
                end
            end
            Enzyme.autodiff(
                fmode, basefunc, BatchDuplicated(y, Jaccache),
                BatchDuplicated(θ, seeds), Const(p)
            )
            for i in eachindex(θ)
                if J isa Vector
                    J[i] = Jaccache[i][1]
                else
                    copyto!(@view(J[:, i]), Jaccache[i])
                end
            end
            # else
            #     Enzyme.autodiff(Enzyme.Reverse, f.cons, BatchDuplicated(y, seeds),
            #         BatchDuplicated(θ, Jaccache), Const(p))
            #     for i in 1:num_cons
            #         if J isa Vector
            #             J .= Jaccache[1]
            #         else
            #             J[i, :] = Jaccache[i]
            #         end
            #     end
            # end
            return
        end
    elseif cons_j == true && cons !== nothing
        cons_j! = (J, θ) -> f.cons_j(J, θ, p)
    else
        cons_j! = nothing
    end

    if cons !== nothing && cons_vjp == true && f.cons_vjp === nothing
        cons_res = zeros(eltype(x), num_cons)
        function cons_vjp!(res, θ, v)
            Enzyme.make_zero!(res)
            Enzyme.make_zero!(cons_res)

            return Enzyme.autodiff(
                rmode,
                f.cons,
                Const,
                Duplicated(cons_res, v),
                Duplicated(θ, res),
                Const(p)
            )
        end
    elseif cons_vjp == true && cons !== nothing
        cons_vjp! = (Jv, θ, σ) -> f.cons_vjp(Jv, θ, σ, p)
    else
        cons_vjp! = nothing
    end

    if cons !== nothing && cons_jvp == true && f.cons_jvp === nothing
        cons_res = zeros(eltype(x), num_cons)

        function cons_jvp!(res, θ, v)
            Enzyme.make_zero!(res)
            Enzyme.make_zero!(cons_res)

            return Enzyme.autodiff(
                fmode,
                f.cons,
                Duplicated(cons_res, res),
                Duplicated(θ, v),
                Const(p)
            )
        end
    elseif cons_jvp == true && cons !== nothing
        cons_jvp! = (Jv, θ, v) -> f.cons_jvp(Jv, θ, v, p)
    else
        cons_jvp! = nothing
    end

    if cons !== nothing && cons_h == true && f.cons_h === nothing
        cons_vdθ = Tuple((Array(r) for r in eachrow(I(length(x)) * one(eltype(x)))))
        cons_bθ = zeros(eltype(x), length(x))
        cons_vdbθ = Tuple(zeros(eltype(x), length(x)) for i in eachindex(x))

        function cons_h!(res, θ)
            for i in 1:num_cons
                Enzyme.make_zero!(cons_bθ)
                Enzyme.make_zero!.(cons_vdbθ)
                Enzyme.autodiff(
                    fmode,
                    cons_f2,
                    Const(rmode),
                    Enzyme.BatchDuplicated(θ, cons_vdθ),
                    Enzyme.BatchDuplicated(cons_bθ, cons_vdbθ),
                    Const(f.cons),
                    Const(p),
                    Const(num_cons),
                    Const(i)
                )

                for j in eachindex(θ)
                    res[i][j, :] .= cons_vdbθ[j]
                end
            end
            return
        end
    elseif cons !== nothing && cons_h == true
        cons_h! = (res, θ) -> f.cons_h(res, θ, p)
    else
        cons_h! = nothing
    end

    if lag_h == true && f.lag_h === nothing && cons !== nothing
        lag_vdθ = Tuple((Array(r) for r in eachrow(I(length(x)) * one(eltype(x)))))
        lag_bθ = zeros(eltype(x), length(x))

        if f.hess_prototype === nothing
            lag_vdbθ = Tuple(zeros(eltype(x), length(x)) for i in eachindex(x))
        else
            #useless right now, looks like there is no way to tell Enzyme the sparsity pattern?
            lag_vdbθ = Tuple((copy(r) for r in eachrow(f.hess_prototype)))
        end

        function lag_h!(h, θ, σ, μ, p = p)
            Enzyme.make_zero!(lag_bθ)
            Enzyme.make_zero!.(lag_vdbθ)

            Enzyme.autodiff(
                fmode,
                lag_grad,
                Const(rmode),
                Enzyme.BatchDuplicated(θ, lag_vdθ),
                Enzyme.BatchDuplicatedNoNeed(lag_bθ, lag_vdbθ),
                Const(lagrangian),
                Const(f.f),
                Const(f.cons),
                Const(p),
                Const(σ),
                Const(μ)
            )
            k = 0

            for i in eachindex(θ)
                vec_lagv = lag_vdbθ[i]
                h[(k + 1):(k + i)] .= @view(vec_lagv[1:i])
                k += i
            end
            return
        end

        function lag_h!(H::AbstractMatrix, θ, σ, μ, p = p)
            Enzyme.make_zero!(H)
            Enzyme.make_zero!(lag_bθ)
            Enzyme.make_zero!.(lag_vdbθ)

            Enzyme.autodiff(
                fmode,
                lag_grad,
                Const(rmode),
                Enzyme.BatchDuplicated(θ, lag_vdθ),
                Enzyme.BatchDuplicatedNoNeed(lag_bθ, lag_vdbθ),
                Const(lagrangian),
                Const(f.f),
                Const(f.cons),
                Const(p),
                Const(σ),
                Const(μ)
            )

            for i in eachindex(θ)
                H[i, :] .= lag_vdbθ[i]
            end
            return
        end
    elseif lag_h == true && cons !== nothing
        lag_h! = (θ, σ, μ, p = p) -> f.lag_h(θ, σ, μ, p)
    else
        lag_h! = nothing
    end

    return OptimizationFunction{true}(
        f.f, adtype;
        grad = grad, fg = fg!, fgh = fgh!,
        hess = hess, hv = hv!,
        cons = cons, cons_j = cons_j!,
        cons_jvp = cons_jvp!, cons_vjp = cons_vjp!,
        cons_h = cons_h!,
        hess_prototype = f.hess_prototype,
        cons_jac_prototype = f.cons_jac_prototype,
        cons_hess_prototype = f.cons_hess_prototype,
        lag_h = lag_h!,
        lag_hess_prototype = f.lag_hess_prototype,
        sys = f.sys,
        expr = f.expr,
        cons_expr = f.cons_expr
    )
end

function OptimizationBase.instantiate_function(
        f::OptimizationFunction{true},
        cache::OptimizationBase.ReInitCache,
        adtype::AutoEnzyme,
        num_cons = 0; kwargs...
    )
    p = cache.p
    x = cache.u0

    return OptimizationBase.instantiate_function(f, x, adtype, p, num_cons; kwargs...)
end

function OptimizationBase.instantiate_function(
        f::OptimizationFunction{false}, x,
        adtype::AutoEnzyme, p, num_cons = 0;
        g = false, h = false, hv = false, fg = false, fgh = false,
        cons_j = false, cons_vjp = false, cons_jvp = false, cons_h = false,
        lag_h = false
    )
    rmode = if adtype.mode isa Nothing
        Enzyme.Reverse
    else
        set_runtime_activity2(Enzyme.Reverse, adtype.mode)
    end

    fmode = if adtype.mode isa Nothing
        Enzyme.Forward
    else
        set_runtime_activity2(Enzyme.Forward, adtype.mode)
    end

    if g == true && f.grad === nothing
        res = zeros(eltype(x), size(x))
        function grad(θ, p = p)
            Enzyme.make_zero!(res)
            Enzyme.autodiff(
                rmode,
                Const(firstapply),
                Active,
                Const(f.f),
                Enzyme.Duplicated(θ, res),
                Const(p)
            )
            return res
        end
    elseif fg == true
        grad = (θ, p = p) -> f.grad(θ, p)
    else
        grad = nothing
    end

    if fg == true && f.fg === nothing
        res_fg = zeros(eltype(x), size(x))
        function fg!(θ, p = p)
            Enzyme.make_zero!(res_fg)
            y = Enzyme.autodiff(
                WithPrimal(rmode),
                Const(firstapply),
                Active,
                Const(f.f),
                Enzyme.Duplicated(θ, res_fg),
                Const(p)
            )[2]
            return y, res
        end
    elseif fg == true
        fg! = (θ, p = p) -> f.fg(θ, p)
    else
        fg! = nothing
    end

    if h == true && f.hess === nothing
        vdθ = Tuple((Array(r) for r in eachrow(I(length(x)) * one(eltype(x)))))
        bθ = zeros(eltype(x), length(x))
        vdbθ = Tuple(zeros(eltype(x), length(x)) for i in eachindex(x))

        function hess(θ, p = p)
            Enzyme.make_zero!(bθ)
            Enzyme.make_zero!.(vdbθ)

            Enzyme.autodiff(
                fmode,
                inner_grad,
                Const(rmode),
                Enzyme.BatchDuplicated(θ, vdθ),
                Enzyme.BatchDuplicated(bθ, vdbθ),
                Const(f.f),
                Const(p)
            )

            return reduce(
                vcat, [reshape(vdbθ[i], (1, length(vdbθ[i]))) for i in eachindex(θ)]
            )
        end
    elseif h == true
        hess = (θ, p = p) -> f.hess(θ, p)
    else
        hess = nothing
    end

    if fgh == true && f.fgh === nothing
        vdθ_fgh = Tuple((Array(r) for r in eachrow(I(length(x)) * one(eltype(x)))))
        vdbθ_fgh = Tuple(zeros(eltype(x), length(x)) for i in eachindex(x))
        G_fgh = zeros(eltype(x), length(x))
        H_fgh = zeros(eltype(x), length(x), length(x))

        function fgh!(θ, p = p)
            Enzyme.make_zero!(G_fgh)
            Enzyme.make_zero!(H_fgh)
            Enzyme.make_zero!.(vdbθ_fgh)

            Enzyme.autodiff(
                fmode,
                inner_grad,
                Const(rmode),
                Enzyme.BatchDuplicated(θ, vdθ_fgh),
                Enzyme.BatchDuplicatedNoNeed(G_fgh, vdbθ_fgh),
                Const(f.f),
                Const(p)
            )

            for i in eachindex(θ)
                H_fgh[i, :] .= vdbθ_fgh[i]
            end
            return G_fgh, H_fgh
        end
    elseif fgh == true
        fgh! = (θ, p = p) -> f.fgh(θ, p)
    else
        fgh! = nothing
    end

    if hv == true && f.hv === nothing
        H = zero(x)
        function hv!(θ, v, p = p)
            dθ = zero(θ)
            Enzyme.make_zero!(H)
            Enzyme.autodiff(
                fmode,
                inner_grad,
                Const(rmode),
                Duplicated(θ, v),
                Duplicated(dθ, H),
                Const(f.f),
                Const(p)
            )
            return H
        end
    elseif hv == true
        hv! = (θ, v, p = p) -> f.hv(θ, v, p)
    else
        hv! = f.hv
    end

    if f.cons === nothing
        cons = nothing
    else
        function cons(θ)
            return f.cons(θ, p)
        end
    end

    if cons_j == true && cons !== nothing && f.cons_j === nothing
        seeds = Enzyme.onehot(x)
        Jaccache = Tuple(zeros(eltype(x), num_cons) for i in 1:length(x))

        function cons_j!(θ)
            for i in eachindex(Jaccache)
                Enzyme.make_zero!(Jaccache[i])
            end
            Jaccache,
                y = Enzyme.autodiff(
                WithPrimal(fmode), f.cons, Duplicated,
                BatchDuplicated(θ, seeds), Const(p)
            )
            if size(y, 1) == 1
                return reduce(vcat, Jaccache)
            else
                return reduce(hcat, Jaccache)
            end
        end
    elseif cons_j == true && cons !== nothing
        cons_j! = (θ) -> f.cons_j(θ, p)
    else
        cons_j! = nothing
    end

    if cons_vjp == true && cons !== nothing && f.cons_vjp == true
        res_vjp = zeros(eltype(x), size(x))
        cons_vjp_res = zeros(eltype(x), num_cons)

        function cons_vjp!(θ, v)
            Enzyme.make_zero!(res_vjp)
            Enzyme.make_zero!(cons_vjp_res)

            Enzyme.autodiff(
                WithPrimal(rmode),
                f.cons,
                Const,
                Duplicated(cons_vjp_res, v),
                Duplicated(θ, res_vjp),
                Const(p)
            )
            return res_vjp
        end
    elseif cons_vjp == true && cons !== nothing
        cons_vjp! = (θ, v) -> f.cons_vjp(θ, v, p)
    else
        cons_vjp! = nothing
    end

    if cons_jvp == true && cons !== nothing && f.cons_jvp == true
        res_jvp = zeros(eltype(x), size(x))
        cons_jvp_res = zeros(eltype(x), num_cons)

        function cons_jvp!(θ, v)
            Enzyme.make_zero!(res_jvp)
            Enzyme.make_zero!(cons_jvp_res)

            Enzyme.autodiff(
                fmode,
                f.cons,
                Duplicated(cons_jvp_res, res_jvp),
                Duplicated(θ, v),
                Const(p)
            )
            return res_jvp
        end
    elseif cons_jvp == true && cons !== nothing
        cons_jvp! = (θ, v) -> f.cons_jvp(θ, v, p)
    else
        cons_jvp! = nothing
    end

    if cons_h == true && cons !== nothing && f.cons_h === nothing
        cons_vdθ = Tuple((Array(r) for r in eachrow(I(length(x)) * one(eltype(x)))))
        cons_bθ = zeros(eltype(x), length(x))
        cons_vdbθ = Tuple(zeros(eltype(x), length(x)) for i in eachindex(x))

        function cons_h!(θ)
            return map(1:num_cons) do i
                Enzyme.make_zero!(cons_bθ)
                Enzyme.make_zero!.(cons_vdbθ)
                Enzyme.autodiff(
                    fmode,
                    cons_f2_oop,
                    Const(rmode),
                    Enzyme.BatchDuplicated(θ, cons_vdθ),
                    Enzyme.BatchDuplicated(cons_bθ, cons_vdbθ),
                    Const(f.cons),
                    Const(p),
                    Const(i)
                )

                return reduce(hcat, cons_vdbθ)
            end
        end
    elseif cons_h == true && cons !== nothing
        cons_h! = (θ) -> f.cons_h(θ, p)
    else
        cons_h! = nothing
    end

    if lag_h == true && f.lag_h === nothing && cons !== nothing
        lag_vdθ = Tuple((Array(r) for r in eachrow(I(length(x)) * one(eltype(x)))))
        lag_bθ = zeros(eltype(x), length(x))
        if f.hess_prototype === nothing
            lag_vdbθ = Tuple(zeros(eltype(x), length(x)) for i in eachindex(x))
        else
            lag_vdbθ = Tuple((copy(r) for r in eachrow(f.hess_prototype)))
        end

        function lag_h!(θ, σ, μ, p = p)
            Enzyme.make_zero!(lag_bθ)
            Enzyme.make_zero!.(lag_vdbθ)

            Enzyme.autodiff(
                fmode,
                lag_grad,
                Const(rmode),
                Enzyme.BatchDuplicated(θ, lag_vdθ),
                Enzyme.BatchDuplicatedNoNeed(lag_bθ, lag_vdbθ),
                Const(lagrangian),
                Const(f.f),
                Const(f.cons),
                Const(p),
                Const(σ),
                Const(μ)
            )

            k = 0

            for i in eachindex(θ)
                vec_lagv = lag_vdbθ[i]
                res[(k + 1):(k + i), :] .= @view(vec_lagv[1:i])
                k += i
            end
            return res
        end
    elseif lag_h == true && cons !== nothing
        lag_h! = (θ, σ, μ, p = p) -> f.lag_h(θ, σ, μ, p)
    else
        lag_h! = nothing
    end

    return OptimizationFunction{false}(
        f.f, adtype; grad = grad,
        fg = fg!, fgh = fgh!,
        hess = hess, hv = hv!,
        cons = cons, cons_j = cons_j!,
        cons_jvp = cons_jvp!, cons_vjp = cons_vjp!,
        cons_h = cons_h!,
        hess_prototype = f.hess_prototype,
        cons_jac_prototype = f.cons_jac_prototype,
        cons_hess_prototype = f.cons_hess_prototype,
        lag_h = lag_h!,
        lag_hess_prototype = f.lag_hess_prototype,
        sys = f.sys,
        expr = f.expr,
        cons_expr = f.cons_expr
    )
end

function OptimizationBase.instantiate_function(
        f::OptimizationFunction{false},
        cache::OptimizationBase.ReInitCache,
        adtype::AutoEnzyme,
        num_cons = 0; kwargs...
    )
    p = cache.p
    x = cache.u0

    return OptimizationBase.instantiate_function(f, x, adtype, p, num_cons; kwargs...)
end

function Enzyme.EnzymeRules.augmented_primal(
    config::Enzyme.EnzymeRules.RevConfigWidth{1},
    func::Const{typeof(OptimizationBase.solve_up)}, ::Type{Duplicated{RT}}, prob,
    sensealg::Unions{
        Const{Nothing}, Const{<:SciMLBase.AbstractSensitivityAlgorithm}
    },
    u0, p, args...; kwargs...
) where {RT}

    @inline function copy_or_reuse(val, idx)
        if Enzyme.EnzymeRules.overwritten(config)[idx] && ismutable(val)
            return deepcopy(val)
        else
            return val
        end
    end

    @inline function arg_copy(i)
        return copy_or_reuse(args[i].val, i + 5)
    end

    res = OptimizationBase._solve_adjoint(
        copy_or_reuse(prob.val, 2), copy_or_reuse(sensealg.val, 3),
        copy_or_reuse(u0.val, 4), copy_or_reuse(p.val, 5),
        SciMLBase.EnzymeOriginator(), ntuple(arg_copy, Val(length(args)))...;
        kwargs...
    )

    dres = Enzyme.make_zero(res[1])::RT
    tup = (dres, res[2])
    return Enzyme.EnzymeRules.AugmentedReturn{RT,RT,Any}(res[1], dres, tup::Any)
end

function Enzyme.EnzymeRules.reverse(
    config::Enzyme.EnzymeRules.RevConfigWidth{1},
    func::Const{typeof(OptimizationBase.solve_up)}, ::Type{Duplicated{RT}}, tape, prob, 
    sensealg::Union{
        Const{Nothing}, Const{<:SciMLBase.AbstractSensitivityAlgorithm},
    },
    u0, p, args...; kwargs...
    ) where {RT}
    dres, clos = tape
    dres = dres::RT
    dargs = clos(dres)

    for (darg, ptr) in zip(dargs, (fun, prob, sensealg, u0, p, args...))
        if ptr isa Enzyme.Const
            continue
        end
        if darg == ChainRulesCore.NoTangent()
            continue
        end
        ptr.dval .+= darg 
    end
    Enzyme.make_zero!(dres.u)
    return ntuple(_->nothing, Val(length(args) + 4))
end

end
