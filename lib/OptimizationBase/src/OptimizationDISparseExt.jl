using OptimizationBase
import OptimizationBase.ArrayInterface
import SciMLBase: OptimizationFunction
import OptimizationBase.LinearAlgebra: I, mul!
import DifferentiationInterface
import DifferentiationInterface: prepare_gradient, prepare_hessian, prepare_hvp,
    prepare_jacobian, value_and_gradient!,
    value_derivative_and_second_derivative!,
    value_and_gradient, value_derivative_and_second_derivative,
    gradient!, hessian!, hvp!, jacobian!, gradient, hessian,
    hvp, jacobian
using ADTypes: ADTypes, AbstractADType
using SparseConnectivityTracer: SparseConnectivityTracer, TracerSparsityDetector
using SparseMatrixColorings: SparseMatrixColorings, GreedyColoringAlgorithm

function instantiate_function(
        f::OptimizationFunction{true}, x, adtype::ADTypes.AutoSparse{<:AbstractADType},
        p = SciMLBase.NullParameters(), num_cons = 0;
        g = false, h = false, hv = false, fg = false, fgh = false,
        cons_j = false, cons_vjp = false, cons_jvp = false, cons_h = false,
        lag_h = false
    )
    adtype, soadtype = generate_sparse_adtype(adtype)
    Tx0 = typeof(x)
    Tp0 = typeof(p)

    if g == true && f.grad === nothing
        prep_grad = prepare_gradient(f.f, adtype.dense_ad, x, Constant(p))
        function grad(res, θ)
            return gradient!(f.f, res, prep_grad, adtype.dense_ad, θ, Constant(p))
        end
        if p !== SciMLBase.NullParameters()
            function grad(res, θ, p)
                return gradient!(f.f, res, prep_grad, adtype.dense_ad, θ, Constant(p))
            end
        end
    elseif g == true
        grad = (G, θ, p = p) -> f.grad(G, θ, p)
    else
        grad = nothing
    end

    # A user-supplied `f.grad` is authoritative: building an AD `fg!` off `f.f` would silently
    # discard it (and any tuned preparation behind it) for every value+gradient evaluation.
    # Same rule as the dense `instantiate_function` (OptimizationDIExt.jl).
    if fg == true && f.fg === nothing && f.grad !== nothing
        fg! = let f = f, p = p
            function (res, θ, p = p)
                f.grad(res, θ, p)
                return f.f(θ, p)
            end
        end
    elseif fg == true && f.fg === nothing
        if g == false
            prep_grad = prepare_gradient(f.f, adtype.dense_ad, x, Constant(p))
        end
        function fg!(res, θ)
            (
                y,
                _,
            ) = value_and_gradient!(
                f.f, res, prep_grad, adtype.dense_ad, θ, Constant(p)
            )
            return y
        end
        if p !== SciMLBase.NullParameters()
            prep_grad = prepare_gradient(f.f, adtype.dense_ad, x, Constant(p))
            function fg!(res, θ, p)
                (
                    y,
                    _,
                ) = value_and_gradient!(
                    f.f, res, prep_grad, adtype.dense_ad, θ, Constant(p)
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
    if f.hess === nothing && h == true
        prep_hess = prepare_hessian(f.f, soadtype, x, Constant(p))
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
                f.f, G, H, prep_hess, soadtype.dense_ad, θ, Constant(p)
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
                    f.f, G, H, prep_hess, soadtype.dense_ad, θ, Constant(p)
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
            return only(hvp!(f.f, (H,), prep_hvp, soadtype.dense_ad, θ, (v,), Constant(p)))
        end
        if p !== SciMLBase.NullParameters() && p !== nothing
            function hv!(H, θ, v, p)
                return only(hvp!(f.f, (H,), prep_hvp, soadtype.dense_ad, θ, (v,), Constant(p)))
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
        cons = let f = f, p = p
            (res, θ, p_call = p) -> f.cons(res, θ, p_call)
        end
    end

    function cons_oop(x)
        _res = zeros(eltype(x), num_cons)
        f.cons(_res, x, p)
        return _res
    end

    function cons_oop(x, i)
        _res = zeros(eltype(x), num_cons)
        f.cons(_res, x, p)
        return _res[i]
    end

    function lagrangian(θ, σ, λ, p)
        if eltype(θ) <: SparseConnectivityTracer.AbstractTracer || !iszero(θ)
            return σ * f.f(θ, p) + dot(λ, cons_oop(θ))
        else
            return dot(λ, cons_oop(θ))
        end
    end

    cons_jac_prototype = f.cons_jac_prototype
    cons_jac_colorvec = f.cons_jac_colorvec
    # The sparse Jacobian prep (sparsity detection + coloring) is the expensive part of this
    # function, so it is built once here and shared between `cons_j!` and the products the
    # backend has no native mode for (see cons_products.jl). A user-supplied `cons_j` takes
    # precedence over differentiating `f.cons` for all three.
    _need_cons_jac = f.cons !== nothing && cons_j == true && f.cons_j === nothing
    _ad_jac_vjp = cons_vjp == true && f.cons !== nothing && f.cons_vjp === nothing &&
        f.cons_j === nothing && !_native_vjp(adtype)
    _ad_jac_jvp = cons_jvp == true && f.cons !== nothing && f.cons_jvp === nothing &&
        f.cons_j === nothing && !_native_jvp(adtype)
    _any_ad_jac = _need_cons_jac || _ad_jac_vjp || _ad_jac_jvp
    cons_oop_p = if _any_ad_jac
        let f = f, num_cons = num_cons
            function (x, p)
                res = Vector{_cons_out_eltype(x, p)}(undef, num_cons)
                f.cons(res, x, p)
                return res
            end
        end
    else
        nothing
    end
    prep_jac = _any_ad_jac ? prepare_jacobian(cons_oop_p, adtype, x, Constant(p)) : nothing
    # Fills `J` at `θ` and the construction `p`, for the products built through the Jacobian.
    # They keep the coloring, which the pullback/pushforward paths below have to drop by
    # falling back to `adtype.dense_ad`.
    ad_cons_jac! = if _ad_jac_vjp || _ad_jac_jvp
        let cons_oop_p = cons_oop_p, prep_jac = prep_jac, adtype = adtype, p = p, Tx0 = Tx0
            function (J, θ)
                return if _prep_valid(Tx0, θ)
                    jacobian!(cons_oop_p, J, prep_jac, adtype, θ, Constant(p))
                else
                    jacobian!(cons_oop_p, J, adtype, θ, Constant(p))
                end
            end
        end
    else
        nothing
    end
    # `jacobian!` decompresses into a buffer carrying the detected pattern.
    ad_jac_buffer() = fill!(similar(sparsity_pattern(prep_jac), eltype(x)), zero(eltype(x)))

    if _need_cons_jac
        cons_j! = let cons_oop_p = cons_oop_p, prep_jac = prep_jac,
                adtype = adtype, p = p, Tx0 = Tx0, Tp0 = Tp0
            function (J, θ, p = p)
                if _prep_valid(Tx0, θ) && _prep_valid(Tp0, p)
                    jacobian!(cons_oop_p, J, prep_jac, adtype, θ, Constant(p))
                else
                    jacobian!(cons_oop_p, J, adtype, θ, Constant(p))
                end
                return size(J, 1) == 1 ? vec(J) : J
            end
        end
        cons_jac_prototype = prep_jac.coloring_result.A
        cons_jac_colorvec = prep_jac.coloring_result.color
    elseif cons_j === true && f.cons !== nothing
        cons_j! = let f = f, p = p
            (J, θ, p = p) -> f.cons_j(J, θ, p)
        end
    else
        cons_j! = nothing
    end

    cons_vjp! = if cons_vjp != true || f.cons === nothing
        nothing
    elseif f.cons_vjp !== nothing
        let f = f, p = p
            (J, θ, v) -> f.cons_vjp(J, θ, v, p)
        end
    elseif f.cons_j !== nothing
        _user_cons_vjp(f, x, p, num_cons)
    elseif _ad_jac_vjp
        _cons_vjp_through_jacobian(ad_cons_jac!, ad_jac_buffer())
    else
        prep_pullback = prepare_pullback(
            cons_oop, adtype.dense_ad, x, (ones(eltype(x), num_cons),)
        )
        let cons_oop = cons_oop, prep_pullback = prep_pullback, adtype = adtype
            (J, θ, v) -> only(pullback!(cons_oop, (J,), prep_pullback, adtype.dense_ad, θ, (v,)))
        end
    end

    cons_jvp! = if cons_jvp != true || f.cons === nothing
        nothing
    elseif f.cons_jvp !== nothing
        let f = f, p = p
            (J, θ, v) -> f.cons_jvp(J, θ, v, p)
        end
    elseif f.cons_j !== nothing
        _user_cons_jvp(f, x, p, num_cons)
    elseif _ad_jac_jvp
        _cons_jvp_through_jacobian(ad_cons_jac!, ad_jac_buffer())
    else
        prep_pushforward = prepare_pushforward(
            cons_oop, adtype.dense_ad, x, (ones(eltype(x), length(x)),)
        )
        let cons_oop = cons_oop, prep_pushforward = prep_pushforward, adtype = adtype
            (J, θ, v) -> only(pushforward!(cons_oop, (J,), prep_pushforward, adtype.dense_ad, θ, (v,)))
        end
    end
    conshess_sparsity = f.cons_hess_prototype
    conshess_colors = f.cons_hess_colorvec
    if f.cons !== nothing && f.cons_h === nothing && cons_h == true
        prep_cons_hess = [
            prepare_hessian(cons_oop, soadtype, x, Constant(i))
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
    elseif cons_h == true && f.cons !== nothing
        cons_h! = (res, θ) -> f.cons_h(res, θ, p)
    else
        cons_h! = nothing
    end

    lag_hess_prototype = f.lag_hess_prototype
    lag_hess_colors = f.lag_hess_colorvec
    if f.cons !== nothing && lag_h == true && f.lag_h === nothing
        lag_prep = prepare_hessian(
            lagrangian, soadtype, x, Constant(one(eltype(x))),
            Constant(ones(eltype(x), num_cons)), Constant(p)
        )
        lag_hess_prototype = lag_prep.coloring_result.A
        lag_hess_colors = lag_prep.coloring_result.color

        function lag_h!(H::AbstractMatrix, θ, σ, λ)
            return if σ == zero(eltype(θ))
                cons_h!(H, θ)
                H *= λ
            else
                hessian!(
                    lagrangian, H, lag_prep, soadtype, θ,
                    Constant(σ), Constant(λ), Constant(p)
                )
            end
        end

        function lag_h!(h, θ, σ, λ)
            H = hessian(
                lagrangian, lag_prep, soadtype, θ, Constant(σ), Constant(λ), Constant(p)
            )
            k = 0
            rows, cols, _ = findnz(H)
            for (i, j) in zip(rows, cols)
                if i <= j
                    k += 1
                    h[k] = H[i, j]
                end
            end
            return
        end

        if p !== SciMLBase.NullParameters() && p !== nothing
            function lag_h!(H::AbstractMatrix, θ, σ, λ, p)
                return if σ == zero(eltype(θ))
                    cons_h(H, θ)
                    H *= λ
                else
                    hessian!(
                        lagrangian, H, lag_prep, soadtype, θ,
                        Constant(σ), Constant(λ), Constant(p)
                    )
                end
            end

            function lag_h!(h, θ, σ, λ, p)
                H = hessian(
                    lagrangian, lag_prep, soadtype, θ,
                    Constant(σ), Constant(λ), Constant(p)
                )
                k = 0
                rows, cols, _ = findnz(H)
                for (i, j) in zip(rows, cols)
                    if i <= j
                        k += 1
                        h[k] = H[i, j]
                    end
                end
                return
            end
        end
    elseif lag_h == true
        lag_h! = (H, θ, σ, λ, p = p) -> f.lag_h(H, θ, σ, λ, p)
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
        lag_hess_colorvec = lag_hess_colors,
        sys = f.sys,
        expr = f.expr,
        cons_expr = f.cons_expr
    )
end

function instantiate_function(
        f::OptimizationFunction{true}, cache::OptimizationBase.ReInitCache,
        adtype::ADTypes.AutoSparse{<:AbstractADType}, num_cons = 0; kwargs...
    )
    x = cache.u0
    p = cache.p

    return instantiate_function(f, x, adtype, p, num_cons; kwargs...)
end

function instantiate_function(
        f::OptimizationFunction{false}, x, adtype::ADTypes.AutoSparse{<:AbstractADType},
        p = SciMLBase.NullParameters(), num_cons = 0;
        g = false, h = false, hv = false, fg = false, fgh = false,
        cons_j = false, cons_vjp = false, cons_jvp = false, cons_h = false,
        lag_h = false
    )
    adtype, soadtype = generate_sparse_adtype(adtype)
    Tx0 = typeof(x)
    Tp0 = typeof(p)

    if g == true && f.grad === nothing
        prep_grad = prepare_gradient(f.f, adtype.dense_ad, x, Constant(p))
        function grad(θ)
            return gradient(f.f, prep_grad, adtype.dense_ad, θ, Constant(p))
        end
        if p !== SciMLBase.NullParameters() && p !== nothing
            function grad(θ, p)
                return gradient(f.f, prep_grad, adtype.dense_ad, θ, Constant(p))
            end
        end
    elseif g == true
        grad = (θ, p = p) -> f.grad(θ, p)
    else
        grad = nothing
    end

    # A user-supplied `f.grad` is authoritative (see the in-place method above).
    if fg == true && f.fg === nothing && f.grad !== nothing
        fg! = let f = f, p = p
            (θ, p = p) -> (f.f(θ, p), f.grad(θ, p))
        end
    elseif fg == true && f.fg === nothing
        if g == false
            prep_grad = prepare_gradient(f.f, adtype.dense_ad, x, Constant(p))
        end
        function fg!(θ)
            (y, G) = value_and_gradient(f.f, prep_grad, adtype.dense_ad, θ, Constant(p))
            return y, G
        end
        if p !== SciMLBase.NullParameters() && p !== nothing
            function fg!(θ, p)
                (y, G) = value_and_gradient(f.f, prep_grad, adtype.dense_ad, θ, Constant(p))
                return y, G
            end
        end
    elseif fg == true
        fg! = (θ, p = p) -> f.fg(θ, p)
    else
        fg! = nothing
    end

    if fgh == true && f.fgh === nothing
        function fgh!(θ)
            (
                y,
                G,
                H,
            ) = value_derivative_and_second_derivative(
                f.f, prep_hess, soadtype, θ, Constant(p)
            )
            return y, G, H
        end

        if p !== SciMLBase.NullParameters() && p !== nothing
            function fgh!(θ, p)
                (
                    y,
                    G,
                    H,
                ) = value_derivative_and_second_derivative(
                    f.f, prep_hess, soadtype, θ, Constant(p)
                )
                return y, G, H
            end
        end
    elseif fgh == true
        fgh! = (θ, p = p) -> f.fgh(θ, p)
    else
        fgh! = nothing
    end

    hess_sparsity = f.hess_prototype
    hess_colors = f.hess_colorvec
    if h == true && f.hess === nothing
        prep_hess = prepare_hessian(f.f, soadtype, x, Constant(p))
        function hess(θ)
            return hessian(f.f, prep_hess, soadtype, θ, Constant(p))
        end
        hess_sparsity = prep_hess.coloring_result.A
        hess_colors = prep_hess.coloring_result.color

        if p !== SciMLBase.NullParameters() && p !== nothing
            function hess(θ, p)
                return hessian(f.f, prep_hess, soadtype, θ, Constant(p))
            end
        end
    elseif h == true
        hess = (θ, p = p) -> f.hess(θ, p)
    else
        hess = nothing
    end

    if hv == true && f.hv === nothing
        prep_hvp = prepare_hvp(
            f.f, soadtype.dense_ad, x, (zeros(eltype(x), size(x)),), Constant(p)
        )
        function hv!(θ, v)
            return only(hvp(f.f, prep_hvp, soadtype.dense_ad, θ, (v,), Constant(p)))
        end

        if p !== SciMLBase.NullParameters() && p !== nothing
            function hv!(θ, v, p)
                return only(hvp(f.f, prep_hvp, soadtype.dense_ad, θ, (v,), Constant(p)))
            end
        end
    elseif hv == true
        hv! = (θ, v, p = p) -> f.hv(θ, v, p)
    else
        hv! = nothing
    end

    if f.cons === nothing
        cons = nothing
    else
        cons = let f = f, p = p
            (x, p_call = p) -> f.cons(x, p_call)
        end
    end

    function lagrangian(θ, σ, λ, p)
        return σ * f.f(θ, p) + dot(λ, f.cons(θ, p))
    end

    cons_jac_prototype = f.cons_jac_prototype
    cons_jac_colorvec = f.cons_jac_colorvec
    # Share the (expensive) sparse Jacobian prep between `cons_j!` and the products the backend
    # has no native mode for; see the in-place method above.
    _need_cons_jac = f.cons !== nothing && cons_j == true && f.cons_j === nothing
    _ad_jac_vjp = cons_vjp == true && f.cons !== nothing && f.cons_vjp === nothing &&
        f.cons_j === nothing && !_native_vjp(adtype)
    _ad_jac_jvp = cons_jvp == true && f.cons !== nothing && f.cons_jvp === nothing &&
        f.cons_j === nothing && !_native_jvp(adtype)
    prep_jac = if _need_cons_jac || _ad_jac_vjp || _ad_jac_jvp
        prepare_jacobian(f.cons, adtype, x, Constant(p))
    else
        nothing
    end
    # Out-of-place, so `jacobian` allocates a `J` of the right eltype per call.
    ad_cons_jac = let f = f, prep_jac = prep_jac, adtype = adtype, p = p, Tx0 = Tx0
        θ -> _prep_valid(Tx0, θ) ?
            jacobian(f.cons, prep_jac, adtype, θ, Constant(p)) :
            jacobian(f.cons, adtype, θ, Constant(p))
    end
    if _need_cons_jac
        cons_j! = let f = f, prep_jac = prep_jac, adtype = adtype,
                p = p, Tx0 = Tx0, Tp0 = Tp0
            function (θ, p = p)
                J = if _prep_valid(Tx0, θ) && _prep_valid(Tp0, p)
                    jacobian(f.cons, prep_jac, adtype, θ, Constant(p))
                else
                    jacobian(f.cons, adtype, θ, Constant(p))
                end
                return size(J, 1) == 1 ? vec(J) : J
            end
        end
        cons_jac_prototype = prep_jac.coloring_result.A
        cons_jac_colorvec = prep_jac.coloring_result.color
    elseif cons_j === true && f.cons !== nothing
        cons_j! = let f = f, p = p
            (θ, p = p) -> f.cons_j(θ, p)
        end
    else
        cons_j! = nothing
    end

    cons_vjp! = if cons_vjp != true || f.cons === nothing
        nothing
    elseif f.cons_vjp !== nothing
        let f = f, p = p
            (θ, v) -> f.cons_vjp(θ, v, p)
        end
    elseif f.cons_j !== nothing
        _user_cons_vjp(f, x, p, num_cons)
    elseif _ad_jac_vjp
        let ad_cons_jac = ad_cons_jac
            (θ, v) -> transpose(ad_cons_jac(θ)) * v
        end
    else
        prep_pullback = prepare_pullback(
            f.cons, adtype.dense_ad, x, (ones(eltype(x), num_cons),), Constant(p)
        )
        let f = f, prep_pullback = prep_pullback, adtype = adtype, p = p
            (θ, v) -> only(pullback(f.cons, prep_pullback, adtype.dense_ad, θ, (v,), Constant(p)))
        end
    end

    cons_jvp! = if cons_jvp != true || f.cons === nothing
        nothing
    elseif f.cons_jvp !== nothing
        let f = f, p = p
            (θ, v) -> f.cons_jvp(θ, v, p)
        end
    elseif f.cons_j !== nothing
        _user_cons_jvp(f, x, p, num_cons)
    elseif _ad_jac_jvp
        let ad_cons_jac = ad_cons_jac
            (θ, v) -> ad_cons_jac(θ) * v
        end
    else
        prep_pushforward = prepare_pushforward(
            f.cons, adtype.dense_ad, x, (ones(eltype(x), length(x)),), Constant(p)
        )
        let f = f, prep_pushforward = prep_pushforward, adtype = adtype, p = p
            (θ, v) -> only(
                pushforward(f.cons, prep_pushforward, adtype.dense_ad, θ, (v,), Constant(p))
            )
        end
    end
    conshess_sparsity = f.cons_hess_prototype
    conshess_colors = f.cons_hess_colorvec
    if f.cons !== nothing && cons_h == true && f.cons_h === nothing
        function cons_i(x, i)
            return f.cons(x, p)[i]
        end
        prep_cons_hess = [
            prepare_hessian(cons_i, soadtype, x, Constant(i))
                for i in 1:num_cons
        ]

        function cons_h!(θ)
            H = map(1:num_cons) do i
                hessian(cons_i, prep_cons_hess[i], soadtype, θ, Constant(i))
            end
            return H
        end
        colores = getfield.(prep_cons_hess, :coloring_result)
        conshess_sparsity = getfield.(colores, :A)
        conshess_colors = getfield.(colores, :color)
    elseif cons_h == true && f.cons !== nothing
        cons_h! = (res, θ) -> f.cons_h(res, θ, p)
    else
        cons_h! = nothing
    end

    lag_hess_prototype = f.lag_hess_prototype
    lag_hess_colors = f.lag_hess_colorvec
    if f.cons !== nothing && lag_h == true && f.lag_h === nothing
        lag_prep = prepare_hessian(
            lagrangian, soadtype, x, Constant(one(eltype(x))),
            Constant(ones(eltype(x), num_cons)), Constant(p)
        )
        function lag_h!(θ, σ, λ)
            if σ == zero(eltype(θ))
                return λ .* cons_h!(θ)
            else
                hess = hessian(
                    lagrangian, lag_prep, soadtype, θ,
                    Constant(σ), Constant(λ), Constant(p)
                )
                return hess
            end
        end
        lag_hess_prototype = lag_prep.coloring_result.A
        lag_hess_colors = lag_prep.coloring_result.color

        if p !== SciMLBase.NullParameters() && p !== nothing
            function lag_h!(θ, σ, λ, p)
                if σ == zero(eltype(θ))
                    return λ .* cons_h!(θ)
                else
                    hess = hessian(
                        lagrangian, lag_prep, θ, Constant(σ), Constant(λ), Constant(p)
                    )
                    return hess
                end
            end
        end
    elseif lag_h == true && f.cons !== nothing
        lag_h! = (θ, σ, μ, p = p) -> f.lag_h(θ, σ, μ, p)
    else
        lag_h! = nothing
    end
    return OptimizationFunction{false}(
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
        lag_hess_colorvec = lag_hess_colors,
        sys = f.sys,
        expr = f.expr,
        cons_expr = f.cons_expr
    )
end

function instantiate_function(
        f::OptimizationFunction{false}, cache::OptimizationBase.ReInitCache,
        adtype::ADTypes.AutoSparse{<:AbstractADType}, num_cons = 0; kwargs...
    )
    x = cache.u0
    p = cache.p

    return instantiate_function(f, x, adtype, p, num_cons; kwargs...)
end
