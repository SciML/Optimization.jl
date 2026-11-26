module OptimizationZygoteExt

using OptimizationBase, SparseArrays
using OptimizationBase.FastClosures
import OptimizationBase.ArrayInterface
import SciMLBase: OptimizationFunction
import OptimizationBase.LinearAlgebra: I, dot
import DifferentiationInterface
import DifferentiationInterface: prepare_gradient, prepare_hessian, prepare_hvp,
    prepare_pullback, prepare_pushforward, pullback!,
    pushforward!,
    pullback, pushforward,
    prepare_jacobian, value_and_gradient!, value_and_gradient,
    value_derivative_and_second_derivative!,
    value_derivative_and_second_derivative,
    gradient!, hessian!, hvp!, jacobian!, gradient, hessian,
    hvp, jacobian, Constant
using ADTypes, SciMLBase
import Zygote, Zygote.ForwardDiff

# σ f(θ, p) + Σᵢ λᵢ cᵢ(θ, p). At σ = 0 the objective is not evaluated, as it may be undefined
# at θ. Zygote returns `nothing` as the gradient of a function that does not depend on θ (at
# σ = 0 with constraints constant in θ); the zero-weighted `sum(θ)` keeps it an array.
function lagrangian_value(f, num_cons, θ, σ, λ, p)
    c = Zygote.Buffer(θ, num_cons)
    f.cons(c, θ, p)
    value = dot(λ, copy(c)) + zero(eltype(θ)) * sum(θ)
    return iszero(σ) ? value : σ * f.f(θ, p) + value
end

function make_lagrangian(f, num_cons)
    return let f = f, num_cons = num_cons
        (θ, σ, λ, p) -> lagrangian_value(f, num_cons, θ, σ, λ, p)
    end
end

# The vector form of a dense `lag_h`: the lower triangle of `H`, row by row.
function pack_lower_triangle!(h, H)
    k = 0
    for i in axes(H, 1), j in 1:i
        k += 1
        h[k] = H[i, j]
    end
    return nothing
end

# The coordinates the vector form of a sparse `lag_h` writes, in order. Solvers read them from
# `lag_hess_prototype`: `lag_hess_structure` of a sparse prototype, the lower triangle row by
# row of a dense one, which a known sparsity pattern given as a dense matrix makes it.
lag_hess_coordinates(prototype::SparseMatrixCSC) = OptimizationBase.lag_hess_structure(prototype)
function lag_hess_coordinates(prototype::AbstractMatrix)
    n = size(prototype, 1)
    return [i for i in 1:n for j in 1:i], [j for i in 1:n for j in 1:i]
end

function OptimizationBase.instantiate_function(
        f::OptimizationFunction{true}, x,
        adtype::Union{
            ADTypes.AutoZygote,
            DifferentiationInterface.SecondOrder{
                <:ADTypes.AbstractADType, <:ADTypes.AutoZygote,
            },
        },
        p = SciMLBase.NullParameters(), num_cons = 0;
        g = false, h = false, hv = false, fg = false, fgh = false,
        cons_j = false, cons_vjp = false, cons_jvp = false, cons_h = false,
        lag_h = false
    )
    return OptimizationBase._instantiate_requested(
        _instantiate_function, f, x, adtype, p, num_cons,
        (; g, h, hv, fg, fgh, cons_j, cons_vjp, cons_jvp, cons_h, lag_h)
    )
end

function _instantiate_function(
        f::OptimizationFunction{true}, x,
        adtype::Union{
            ADTypes.AutoZygote,
            DifferentiationInterface.SecondOrder{
                <:ADTypes.AbstractADType, <:ADTypes.AutoZygote,
            },
        },
        p, num_cons, ::Val{flags}
    ) where {flags}
    (; g, h, hv, fg, fgh, cons_j, cons_vjp, cons_jvp, cons_h, lag_h) = flags
    adtype, soadtype = OptimizationBase.generate_adtype(adtype)

    if g == true && f.grad === nothing
        prep_grad = prepare_gradient(f.f, adtype, x, Constant(p), strict = Val(false))
        function grad(res, θ)
            return gradient!(f.f, res, prep_grad, adtype, θ, Constant(p))
        end
        if p !== SciMLBase.NullParameters() && p !== nothing
            function grad(res, θ, p)
                return gradient!(f.f, res, prep_grad, adtype, θ, Constant(p))
            end
        end
    elseif g == true
        grad = (G, θ, p = p) -> f.grad(G, θ, p)
    else
        grad = nothing
    end

    if fg == true && f.fg === nothing && f.grad !== nothing
        # A user-supplied gradient is authoritative; building an AD `fg!` off `f.f` would
        # silently discard it for every value+gradient evaluation.
        fg! = (res, θ, p = p) -> (f.grad(res, θ, p); f.f(θ, p))
    elseif fg == true && f.fg === nothing
        # `f.grad === nothing` here, so `g == true` means the branch above already built the prep.
        if g == false
            prep_grad = prepare_gradient(f.f, adtype, x, Constant(p), strict = Val(false))
        end
        function fg!(res, θ)
            (y, _) = value_and_gradient!(f.f, res, prep_grad, adtype, θ, Constant(p))
            return y
        end
        if p !== SciMLBase.NullParameters() && p !== nothing
            function fg!(res, θ, p)
                (y, _) = value_and_gradient!(f.f, res, prep_grad, adtype, θ, Constant(p))
                return y
            end
        end
    elseif fg == true
        fg! = (G, θ, p = p) -> f.fg(G, θ, p)
    else
        fg! = nothing
    end

    hess_sparsity = f.hess_prototype
    hess_colors = f.hess_colorvec
    if h == true && f.hess === nothing
        prep_hess = prepare_hessian(f.f, soadtype, x, Constant(p), strict = Val(false))
        function hess(res, θ)
            return hessian!(f.f, res, prep_hess, soadtype, θ, Constant(p))
        end
        if p !== SciMLBase.NullParameters() && p !== nothing
            function hess(res, θ, p)
                return hessian!(f.f, res, prep_hess, soadtype, θ, Constant(p))
            end
        end
    elseif h == true
        hess = (H, θ, p = p) -> f.hess(H, θ, p)
    else
        hess = nothing
    end

    if fgh == true && f.fgh === nothing
        function fgh!(G, H, θ)
            (
                y,
                _,
                _,
            ) = value_derivative_and_second_derivative!(
                f.f, G, H, prep_hess, soadtype, θ, Constant(p)
            )
            return y
        end
        if p !== SciMLBase.NullParameters() && p !== nothing
            function fgh!(G, H, θ, p)
                (
                    y,
                    _,
                    _,
                ) = value_derivative_and_second_derivative!(
                    f.f, G, H, prep_hess, soadtype, θ, Constant(p)
                )
                return y
            end
        end
    elseif fgh == true
        fgh! = (G, H, θ, p = p) -> f.fgh(G, H, θ, p)
    else
        fgh! = nothing
    end

    if hv == true && f.hv === nothing
        prep_hvp = prepare_hvp(f.f, soadtype, x, (zeros(eltype(x), size(x)),), Constant(p), strict = Val(false))
        function hv!(H, θ, v)
            return hvp!(f.f, (H,), prep_hvp, soadtype, θ, (v,), Constant(p))
        end
        if p !== SciMLBase.NullParameters() && p !== nothing
            function hv!(H, θ, v, p)
                return hvp!(f.f, (H,), prep_hvp, soadtype, θ, (v,), Constant(p))
            end
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

        function cons_oop(x)
            _res = Zygote.Buffer(x, num_cons)
            f.cons(_res, x, p)
            return copy(_res)
        end

        function cons_oop(x, i)
            _res = Zygote.Buffer(x, num_cons)
            f.cons(_res, x, p)
            return _res[i]
        end
    end

    cons_jac_prototype = f.cons_jac_prototype
    cons_jac_colorvec = f.cons_jac_colorvec
    if cons !== nothing && cons_j == true && f.cons_j === nothing
        prep_jac = prepare_jacobian(cons_oop, adtype, x, strict = Val(false))
        function cons_j!(J, θ)
            jacobian!(cons_oop, J, prep_jac, adtype, θ)
            return if size(J, 1) == 1
                J = vec(J)
            end
        end
    elseif cons !== nothing && cons_j == true
        cons_j! = (J, θ) -> f.cons_j(J, θ, p)
    else
        cons_j! = nothing
    end

    if f.cons_vjp === nothing && cons_vjp == true && cons !== nothing
        prep_pullback = prepare_pullback(
            cons_oop, adtype, x, (ones(eltype(x), num_cons),), strict = Val(false)
        )
        function cons_vjp!(J, θ, v)
            return pullback!(cons_oop, (J,), prep_pullback, adtype, θ, (v,))
        end
    elseif cons_vjp == true
        cons_vjp! = (J, θ, v) -> f.cons_vjp(J, θ, v, p)
    else
        cons_vjp! = nothing
    end

    if cons !== nothing && f.cons_jvp === nothing && cons_jvp == true
        prep_pushforward = prepare_pushforward(
            cons_oop, adtype, x, (ones(eltype(x), length(x)),), strict = Val(false)
        )
        function cons_jvp!(J, θ, v)
            return pushforward!(cons_oop, (J,), prep_pushforward, adtype, θ, (v,))
        end
    elseif cons_jvp == true
        cons_jvp! = (J, θ, v) -> f.cons_jvp(J, θ, v, p)
    else
        cons_jvp! = nothing
    end

    conshess_sparsity = f.cons_hess_prototype
    conshess_colors = f.cons_hess_colorvec

    if cons !== nothing && f.cons_h === nothing && cons_h == true
        prep_cons_hess = [
            prepare_hessian(
                cons_oop, soadtype, x, Constant(i), strict = Val(false)
            )
                for i in 1:num_cons
        ]
        cons_h! = function (H, θ)
            for i in 1:num_cons
                hessian!(cons_oop, H[i], prep_cons_hess[i], soadtype, θ, Constant(i))
            end
            return
        end
    elseif cons !== nothing && cons_h == true
        cons_h! = (res, θ) -> f.cons_h(res, θ, p)
    else
        cons_h! = nothing
    end

    lag_hess_prototype = f.lag_hess_prototype

    if f.lag_h === nothing && cons !== nothing && lag_h == true
        lagrangian = make_lagrangian(f, num_cons)
        lag_extras = prepare_hessian(
            lagrangian, soadtype, x, Constant(one(eltype(x))),
            Constant(ones(eltype(x), num_cons)), Constant(p), strict = Val(false)
        )
        lag_hess_prototype = zeros(Bool, length(x), length(x))

        lag_h! = let lagrangian = lagrangian, lag_extras = lag_extras,
                soadtype = soadtype, p = p
            function _lag_h!(H::AbstractMatrix, θ, σ, λ, p = p)
                return hessian!(
                    lagrangian, H, lag_extras, soadtype, θ,
                    Constant(σ), Constant(λ), Constant(p)
                )
            end
            function _lag_h!(h::AbstractVector, θ, σ, λ, p = p)
                H = hessian(
                    lagrangian, lag_extras, soadtype, θ,
                    Constant(σ), Constant(λ), Constant(p)
                )
                return pack_lower_triangle!(h, H)
            end
            _lag_h!
        end
    elseif cons !== nothing && lag_h == true
        lag_h! = (res, θ, σ, μ, p = p) -> f.lag_h(res, θ, σ, μ, p)
    else
        lag_h! = nothing
    end

    return OptimizationFunction{true}(
        f.f, adtype;
        grad = grad, fg = fg!, hess = hess, hv = hv!, fgh = fgh!,
        cons = cons, cons_j = cons_j!, cons_h = cons_h!,
        cons_vjp = cons_vjp!, cons_jvp = cons_jvp!,
        hess_prototype = hess_sparsity,
        hess_colorvec = hess_colors,
        cons_jac_prototype = cons_jac_prototype,
        cons_jac_colorvec = cons_jac_colorvec,
        cons_hess_prototype = conshess_sparsity,
        cons_hess_colorvec = conshess_colors,
        lag_h = lag_h!,
        lag_hess_prototype = lag_hess_prototype,
        sys = f.sys,
        expr = f.expr,
        cons_expr = f.cons_expr
    )
end

function OptimizationBase.instantiate_function(
        f::OptimizationFunction{true}, cache::OptimizationBase.ReInitCache,
        adtype::ADTypes.AutoZygote, num_cons = 0; kwargs...
    )
    x = cache.u0
    p = cache.p

    return OptimizationBase.instantiate_function(
        f, x, adtype, p, num_cons; kwargs...
    )
end

function OptimizationBase.instantiate_function(
        f::OptimizationFunction{true}, cache::OptimizationBase.ReInitCache,
        adtype::DifferentiationInterface.SecondOrder{
            <:ADTypes.AbstractADType, <:ADTypes.AutoZygote,
        },
        num_cons = 0; kwargs...
    )
    x = cache.u0
    p = cache.p

    return OptimizationBase.instantiate_function(
        f, x, adtype, p, num_cons; kwargs...
    )
end

function OptimizationBase.instantiate_function(
        f::OptimizationFunction{true}, x,
        adtype::ADTypes.AutoSparse{
            <:Union{
                ADTypes.AutoZygote,
                DifferentiationInterface.SecondOrder{
                    <:ADTypes.AbstractADType, <:ADTypes.AutoZygote,
                },
            },
        },
        p = SciMLBase.NullParameters(), num_cons = 0;
        g = false, h = false, hv = false, fg = false, fgh = false,
        cons_j = false, cons_vjp = false, cons_jvp = false, cons_h = false,
        lag_h = false
    )
    return OptimizationBase._instantiate_requested(
        _instantiate_function, f, x, adtype, p, num_cons,
        (; g, h, hv, fg, fgh, cons_j, cons_vjp, cons_jvp, cons_h, lag_h)
    )
end

function _instantiate_function(
        f::OptimizationFunction{true}, x,
        adtype::ADTypes.AutoSparse{
            <:Union{
                ADTypes.AutoZygote,
                DifferentiationInterface.SecondOrder{
                    <:ADTypes.AbstractADType, <:ADTypes.AutoZygote,
                },
            },
        },
        p, num_cons, ::Val{flags}
    ) where {flags}
    (; g, h, hv, fg, fgh, cons_j, cons_vjp, cons_jvp, cons_h, lag_h) = flags
    adtype, soadtype = OptimizationBase.generate_sparse_adtype(adtype)

    if g == true && f.grad === nothing
        extras_grad = prepare_gradient(
            f.f, adtype.dense_ad, x, Constant(p), strict = Val(false)
        )
        function grad(res, θ)
            return gradient!(f.f, res, extras_grad, adtype.dense_ad, θ, Constant(p))
        end
        if p !== SciMLBase.NullParameters() && p !== nothing
            function grad(res, θ, p)
                return gradient!(f.f, res, extras_grad, adtype.dense_ad, θ, Constant(p))
            end
        end
    elseif g == true
        grad = (G, θ, p = p) -> f.grad(G, θ, p)
    else
        grad = nothing
    end

    if fg == true && f.fg === nothing && f.grad !== nothing
        # A user-supplied gradient is authoritative; see the dense path above.
        fg! = (res, θ, p = p) -> (f.grad(res, θ, p); f.f(θ, p))
    elseif fg == true && f.fg === nothing
        if g == false
            extras_grad = prepare_gradient(
                f.f, adtype.dense_ad, x, Constant(p), strict = Val(false)
            )
        end
        function fg!(res, θ)
            (
                y,
                _,
            ) = value_and_gradient!(
                f.f, res, extras_grad, adtype.dense_ad, θ, Constant(p)
            )
            return y
        end
        if p !== SciMLBase.NullParameters() && p !== nothing
            function fg!(res, θ, p)
                (
                    y,
                    _,
                ) = value_and_gradient!(
                    f.f, res, extras_grad, adtype.dense_ad, θ, Constant(p)
                )
                return y
            end
        end
    elseif fg == true
        fg! = (G, θ, p = p) -> f.fg(G, θ, p)
    else
        fg! = nothing
    end

    hess_sparsity = f.hess_prototype
    hess_colors = f.hess_colorvec
    if h == true && f.hess === nothing
        prep_hess = prepare_hessian(f.f, soadtype, x, Constant(p), strict = Val(false))
        function hess(res, θ)
            return hessian!(f.f, res, prep_hess, soadtype, θ, Constant(p))
        end
        hess_sparsity = prep_hess.coloring_result.A
        hess_colors = prep_hess.coloring_result.color

        if p !== SciMLBase.NullParameters() && p !== nothing
            function hess(res, θ, p)
                return hessian!(f.f, res, prep_hess, soadtype, θ, Constant(p))
            end
        end
    elseif h == true
        hess = (H, θ, p = p) -> f.hess(H, θ, p)
    else
        hess = nothing
    end

    if fgh == true && f.fgh === nothing
        function fgh!(G, H, θ)
            (
                y,
                _,
                _,
            ) = value_derivative_and_second_derivative!(
                f.f, G, H, θ, prep_hess, soadtype, Constant(p)
            )
            return y
        end

        if p !== SciMLBase.NullParameters() && p !== nothing
            function fgh!(G, H, θ, p)
                (
                    y,
                    _,
                    _,
                ) = value_derivative_and_second_derivative!(
                    f.f, G, H, θ, prep_hess, soadtype, Constant(p)
                )
                return y
            end
        end
    elseif fgh == true
        fgh! = (G, H, θ, p = p) -> f.fgh(G, H, θ, p)
    else
        fgh! = nothing
    end

    if hv == true && f.hv === nothing
        prep_hvp = prepare_hvp(
            f.f, soadtype.dense_ad, x, (zeros(eltype(x), size(x)),), Constant(p)
        )
        function hv!(H, θ, v)
            return hvp!(f.f, (H,), prep_hvp, soadtype.dense_ad, θ, (v,), Constant(p))
        end
        if p !== SciMLBase.NullParameters() && p !== nothing
            function hv!(H, θ, v, p)
                return hvp!(f.f, (H,), prep_hvp, soadtype.dense_ad, θ, (v,), Constant(p))
            end
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

        function cons_oop(x)
            _res = Zygote.Buffer(x, num_cons)
            f.cons(_res, x, p)
            return copy(_res)
        end

        function cons_oop(x, i)
            _res = Zygote.Buffer(x, num_cons)
            f.cons(_res, x, p)
            return _res[i]
        end
    end

    cons_jac_prototype = f.cons_jac_prototype
    cons_jac_colorvec = f.cons_jac_colorvec
    if cons !== nothing && cons_j == true && f.cons_j === nothing
        prep_jac = prepare_jacobian(cons_oop, adtype, x)
        function cons_j!(J, θ)
            jacobian!(cons_oop, J, prep_jac, adtype, θ)
            return if size(J, 1) == 1
                J = vec(J)
            end
        end
        cons_jac_prototype = prep_jac.coloring_result.A
        cons_jac_colorvec = prep_jac.coloring_result.color
    elseif cons !== nothing && cons_j == true
        cons_j! = (J, θ) -> f.cons_j(J, θ, p)
    else
        cons_j! = nothing
    end

    if f.cons_vjp === nothing && cons_vjp == true && cons !== nothing
        extras_pullback = prepare_pullback(
            cons_oop, adtype.dense_ad, x, (ones(eltype(x), num_cons),)
        )
        function cons_vjp!(J, θ, v)
            return pullback!(
                cons_oop, (J,), extras_pullback, adtype.dense_ad, θ, (v,)
            )
        end
    elseif cons_vjp == true
        cons_vjp! = (J, θ, v) -> f.cons_vjp(J, θ, v, p)
    else
        cons_vjp! = nothing
    end

    if f.cons_jvp === nothing && cons_jvp == true && cons !== nothing
        extras_pushforward = prepare_pushforward(
            cons_oop, adtype.dense_ad, x, (ones(eltype(x), length(x)),)
        )
        function cons_jvp!(J, θ, v)
            return pushforward!(
                cons_oop, (J,), extras_pushforward, adtype.dense_ad, θ, (v,)
            )
        end
    elseif cons_jvp == true
        cons_jvp! = (J, θ, v) -> f.cons_jvp(J, θ, v, p)
    else
        cons_jvp! = nothing
    end

    conshess_sparsity = f.cons_hess_prototype
    conshess_colors = f.cons_hess_colorvec
    if cons !== nothing && f.cons_h === nothing && cons_h == true
        prep_cons_hess = [
            prepare_hessian(
                cons_oop, soadtype, x, Constant(i), strict = Val(false)
            )
                for i in 1:num_cons
        ]
        colores = getfield.(prep_cons_hess, :coloring_result)
        conshess_sparsity = getfield.(colores, :A)
        conshess_colors = getfield.(colores, :color)
        function cons_h!(H, θ)
            for i in 1:num_cons
                hessian!(cons_oop, H[i], prep_cons_hess[i], soadtype, θ, Constant(i))
            end
            return
        end
    elseif cons_h == true
        cons_h! = (res, θ) -> f.cons_h(res, θ, p)
    else
        cons_h! = nothing
    end

    lag_hess_prototype = f.lag_hess_prototype
    lag_hess_colors = f.lag_hess_colorvec
    if cons !== nothing && f.lag_h === nothing && lag_h == true
        lagrangian = make_lagrangian(f, num_cons)
        # Prepared at σ = 1, so the pattern also holds the entries of the objective; at σ = 0
        # they are zero.
        lag_extras = prepare_hessian(
            lagrangian, soadtype, x, Constant(one(eltype(x))),
            Constant(ones(eltype(x), num_cons)), Constant(p), strict = Val(false)
        )
        lag_hess_prototype = lag_extras.coloring_result.A
        lag_hess_colors = lag_extras.coloring_result.color
        lag_rows, lag_cols = lag_hess_coordinates(lag_hess_prototype)

        lag_h! = let lagrangian = lagrangian, lag_extras = lag_extras,
                soadtype = soadtype, p = p, prototype = lag_hess_prototype,
                lag_rows = lag_rows, lag_cols = lag_cols
            function _lag_h!(H::AbstractMatrix, θ, σ, λ, p = p)
                return hessian!(
                    lagrangian, H, lag_extras, soadtype, θ,
                    Constant(σ), Constant(λ), Constant(p)
                )
            end
            function _lag_h!(h::AbstractVector, θ, σ, λ, p = p)
                H = similar(prototype, promote_type(eltype(θ), eltype(h)))
                hessian!(
                    lagrangian, H, lag_extras, soadtype, θ,
                    Constant(σ), Constant(λ), Constant(p)
                )
                for k in eachindex(lag_rows, lag_cols)
                    h[k] = H[lag_rows[k], lag_cols[k]]
                end
                return nothing
            end
            _lag_h!
        end
    elseif cons !== nothing && lag_h == true
        lag_h! = (res, θ, σ, μ, p = p) -> f.lag_h(res, θ, σ, μ, p)
    else
        lag_h! = nothing
    end
    return OptimizationFunction{true}(
        f.f, adtype;
        grad = grad, fg = fg!, hess = hess, hv = hv!, fgh = fgh!,
        cons = cons, cons_j = cons_j!, cons_h = cons_h!,
        hess_prototype = hess_sparsity,
        hess_colorvec = hess_colors,
        cons_jac_prototype = cons_jac_prototype,
        cons_jac_colorvec = cons_jac_colorvec,
        cons_hess_prototype = conshess_sparsity,
        cons_hess_colorvec = conshess_colors,
        lag_h = lag_h!,
        lag_hess_prototype = lag_hess_prototype,
        lag_hess_colorvec = lag_hess_colors,
        sys = f.sys,
        expr = f.expr,
        cons_expr = f.cons_expr
    )
end

function OptimizationBase.instantiate_function(
        f::OptimizationFunction{true}, cache::OptimizationBase.ReInitCache,
        adtype::ADTypes.AutoSparse{<:AutoZygote}, num_cons = 0; kwargs...
    )
    x = cache.u0
    p = cache.p

    return OptimizationBase.instantiate_function(f, x, adtype, p, num_cons; kwargs...)
end

function OptimizationBase.instantiate_function(
        f::OptimizationFunction{true}, cache::OptimizationBase.ReInitCache,
        adtype::ADTypes.AutoSparse{
            <:DifferentiationInterface.SecondOrder{
                <:ADTypes.AbstractADType, <:ADTypes.AutoZygote,
            },
        },
        num_cons = 0; kwargs...
    )
    x = cache.u0
    p = cache.p

    return OptimizationBase.instantiate_function(f, x, adtype, p, num_cons; kwargs...)
end

end
