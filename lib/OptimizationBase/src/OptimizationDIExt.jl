using OptimizationBase
import OptimizationBase.ArrayInterface
import SciMLBase: OptimizationFunction
import OptimizationBase.LinearAlgebra: I
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
using OptimizationBase.FastClosures

function instantiate_function(
        f::OptimizationFunction{true}, x, ::ADTypes.AutoSparse{<:ADTypes.AutoSymbolics},
        args...; kwargs...
    )
    return instantiate_function(f, x, ADTypes.AutoSymbolics(), args...; kwargs...)
end
function instantiate_function(
        f::OptimizationFunction{true}, cache::OptimizationBase.ReInitCache,
        ::ADTypes.AutoSparse{<:ADTypes.AutoSymbolics}, args...; kwargs...
    )
    x = cache.u0
    p = cache.p

    return instantiate_function(f, x, ADTypes.AutoSymbolics(), p, args...; kwargs...)
end
function instantiate_function(
        f::OptimizationFunction{true}, x, adtype::ADTypes.AbstractADType,
        p = SciMLBase.NullParameters(), num_cons = 0;
        g = false, h = false, hv = false, fg = false, fgh = false,
        cons_j = false, cons_vjp = false, cons_jvp = false, cons_h = false,
        lag_h = false
    )
    adtype, soadtype = generate_adtype(adtype)

    # Create gradient closures with proper type stability using let blocks
    grad = if g == true && f.grad === nothing
        _prep_grad = prepare_gradient(f.f, adtype, x, Constant(p))
        if p !== SciMLBase.NullParameters() && p !== nothing
            let _prep_grad = _prep_grad, f = f, adtype = adtype
                (res, θ, p = p) -> gradient!(f.f, res, _prep_grad, adtype, θ, Constant(p))
            end
        else
            let _prep_grad = _prep_grad, f = f, adtype = adtype, p = p
                (res, θ, p = p) -> gradient!(f.f, res, _prep_grad, adtype, θ, Constant(p))
            end
        end
    elseif g == true
        (G, θ, p = p) -> f.grad(G, θ, p)
    else
        nothing
    end

    # Create fg! closures - need separate prep if g was false
    fg! = if fg == true && f.fg === nothing
        _prep_grad_fg = if g == false
            prepare_gradient(f.f, adtype, x, Constant(p))
        else
            # Reuse the prep from gradient if available
            prepare_gradient(f.f, adtype, x, Constant(p))
        end
        if p !== SciMLBase.NullParameters() && p !== nothing
            let _prep_grad_fg = _prep_grad_fg, f = f, adtype = adtype
                function (res, θ, p = p)
                    (y, _) = value_and_gradient!(f.f, res, _prep_grad_fg, adtype, θ, Constant(p))
                    return y
                end
            end
        else
            let _prep_grad_fg = _prep_grad_fg, f = f, adtype = adtype, p = p
                function (res, θ, p = p)
                    (y, _) = value_and_gradient!(f.f, res, _prep_grad_fg, adtype, θ, Constant(p))
                    return y
                end
            end
        end
    elseif fg == true
        (G, θ, p = p) -> f.fg(G, θ, p)
    else
        nothing
    end

    hess_sparsity = f.hess_prototype
    hess_colors = f.hess_colorvec

    # Create hessian closures with proper type stability
    _prep_hess = if (h == true && f.hess === nothing) || (fgh == true && f.fgh === nothing)
        prepare_hessian(f.f, soadtype, x, Constant(p))
    else
        nothing
    end

    hess = if h == true && f.hess === nothing
        if p !== SciMLBase.NullParameters() && p !== nothing
            let _prep_hess = _prep_hess, f = f, soadtype = soadtype
                (res, θ, p = p) -> hessian!(f.f, res, _prep_hess, soadtype, θ, Constant(p))
            end
        else
            let _prep_hess = _prep_hess, f = f, soadtype = soadtype, p = p
                (res, θ, p = p) -> hessian!(f.f, res, _prep_hess, soadtype, θ, Constant(p))
            end
        end
    elseif h == true
        (H, θ, p = p) -> f.hess(H, θ, p)
    else
        nothing
    end

    fgh! = if fgh == true && f.fgh === nothing
        if p !== SciMLBase.NullParameters() && p !== nothing
            let _prep_hess = _prep_hess, f = f, soadtype = soadtype
                function (G, H, θ, p = p)
                    (y, _, _) = value_derivative_and_second_derivative!(
                        f.f, G, H, _prep_hess, soadtype, θ, Constant(p)
                    )
                    return y
                end
            end
        else
            let _prep_hess = _prep_hess, f = f, soadtype = soadtype, p = p
                function (G, H, θ, p = p)
                    (y, _, _) = value_derivative_and_second_derivative!(
                        f.f, G, H, _prep_hess, soadtype, θ, Constant(p)
                    )
                    return y
                end
            end
        end
    elseif fgh == true
        (G, H, θ, p = p) -> f.fgh(G, H, θ, p)
    else
        nothing
    end

    hv! = if hv == true && f.hv === nothing
        _prep_hvp = prepare_hvp(f.f, soadtype, x, (zeros(eltype(x), size(x)),), Constant(p))
        if p !== SciMLBase.NullParameters() && p !== nothing
            let _prep_hvp = _prep_hvp, f = f, soadtype = soadtype
                (H, θ, v, p = p) -> only(hvp!(f.f, (H,), _prep_hvp, soadtype, θ, (v,), Constant(p)))
            end
        else
            let _prep_hvp = _prep_hvp, f = f, soadtype = soadtype, p = p
                (H, θ, v, p = p) -> only(hvp!(f.f, (H,), _prep_hvp, soadtype, θ, (v,), Constant(p)))
            end
        end
    elseif hv == true
        (H, θ, v, p = p) -> f.hv(H, θ, v, p)
    else
        nothing
    end

    # Create constraint-related closures with proper type stability
    cons_oop = if f.cons !== nothing
        let f = f, p = p, num_cons = num_cons
            function _cons_oop(x)
                _res = zeros(eltype(x), num_cons)
                f.cons(_res, x, p)
                return _res
            end
            function _cons_oop(x, i)
                _res = zeros(eltype(x), num_cons)
                f.cons(_res, x, p)
                return _res[i]
            end
            _cons_oop
        end
    else
        nothing
    end

    cons = if f.cons !== nothing
        let f = f, p = p
            (res, x) -> f.cons(res, x, p)
        end
    else
        nothing
    end

    lagrangian = if f.cons !== nothing
        let f = f, cons_oop = cons_oop
            (θ, σ, λ, p) -> σ * f.f(θ, p) + dot(λ, cons_oop(θ))
        end
    else
        nothing
    end

    cons_jac_prototype = f.cons_jac_prototype
    cons_jac_colorvec = f.cons_jac_colorvec

    cons_j! = if f.cons !== nothing && cons_j == true && f.cons_j === nothing
        _prep_jac = prepare_jacobian(cons_oop, adtype, x)
        let cons_oop = cons_oop, _prep_jac = _prep_jac, adtype = adtype
            function (J, θ)
                jacobian!(cons_oop, J, _prep_jac, adtype, θ)
                return if size(J, 1) == 1
                    J = vec(J)
                end
            end
        end
    elseif cons_j == true && f.cons !== nothing
        let f = f, p = p
            (J, θ) -> f.cons_j(J, θ, p)
        end
    else
        nothing
    end

    cons_vjp! = if f.cons_vjp === nothing && cons_vjp == true && f.cons !== nothing
        _prep_pullback = prepare_pullback(cons_oop, adtype, x, (ones(eltype(x), num_cons),))
        let cons_oop = cons_oop, _prep_pullback = _prep_pullback, adtype = adtype
            (J, θ, v) -> only(pullback!(cons_oop, (J,), _prep_pullback, adtype, θ, (v,)))
        end
    elseif cons_vjp == true && f.cons !== nothing
        let f = f, p = p
            (J, θ, v) -> f.cons_vjp(J, θ, v, p)
        end
    else
        nothing
    end

    cons_jvp! = if f.cons_jvp === nothing && cons_jvp == true && f.cons !== nothing
        _prep_pushforward = prepare_pushforward(
            cons_oop, adtype, x, (ones(eltype(x), length(x)),)
        )
        let cons_oop = cons_oop, _prep_pushforward = _prep_pushforward, adtype = adtype
            (J, θ, v) -> only(pushforward!(cons_oop, (J,), _prep_pushforward, adtype, θ, (v,)))
        end
    elseif cons_jvp == true && f.cons !== nothing
        let f = f, p = p
            (J, θ, v) -> f.cons_jvp(J, θ, v, p)
        end
    else
        nothing
    end

    conshess_sparsity = f.cons_hess_prototype
    conshess_colors = f.cons_hess_colorvec

    # Prepare constraint Hessian preparations if needed by lag_h or cons_h
    prep_cons_hess = if f.cons !== nothing && f.cons_h === nothing && (cons_h == true || lag_h == true)
        # This is necessary because DI will create a symbolic index for `Constant(i)`
        # to trace into the function, since it assumes `Constant` can change between
        # DI calls.
        if adtype isa ADTypes.AutoSymbolics
            [prepare_hessian(Base.Fix2(cons_oop, i), soadtype, x) for i in 1:num_cons]
        else
            [prepare_hessian(cons_oop, soadtype, x, Constant(i)) for i in 1:num_cons]
        end
    else
        nothing
    end

    # Generate cons_h! functions
    (cons_h!, cons_h_weighted!) = if f.cons !== nothing && f.cons_h === nothing && prep_cons_hess !== nothing
        _cons_h! = if cons_h == true
            if adtype isa ADTypes.AutoSymbolics
                let cons_oop = cons_oop, prep_cons_hess = prep_cons_hess, soadtype = soadtype, num_cons = num_cons
                    function (H, θ)
                        for i in 1:num_cons
                            hessian!(Base.Fix2(cons_oop, i), H[i], prep_cons_hess[i], soadtype, θ)
                        end
                        return
                    end
                end
            else
                let cons_oop = cons_oop, prep_cons_hess = prep_cons_hess, soadtype = soadtype, num_cons = num_cons
                    function (H, θ)
                        for i in 1:num_cons
                            hessian!(cons_oop, H[i], prep_cons_hess[i], soadtype, θ, Constant(i))
                        end
                        return
                    end
                end
            end
        else
            nothing
        end

        # Weighted sum dispatch for cons_h! (always created if prep_cons_hess exists)
        # This is used by lag_h! when σ=0
        _cons_h_weighted! = let cons_oop = cons_oop, prep_cons_hess = prep_cons_hess, soadtype = soadtype, num_cons = num_cons
            function (H::AbstractMatrix, θ, λ)
                # Compute weighted sum: H = Σᵢ λᵢ∇²cᵢ
                H .= zero(eltype(H))

                # Create a single temporary matrix to reuse for all constraints
                Hi = similar(H)

                for i in 1:num_cons
                    if λ[i] != zero(eltype(λ))
                        # Compute constraint's Hessian into temporary matrix
                        hessian!(cons_oop, Hi, prep_cons_hess[i], soadtype, θ, Constant(i))
                        # Add weighted Hessian to result using in-place operation
                        # H += λ[i] * Hi
                        @. H += λ[i] * Hi
                    end
                end
                return
            end
        end
        (_cons_h!, _cons_h_weighted!)
    elseif cons_h == true && f.cons !== nothing
        (
            let f = f, p = p
                (res, θ) -> f.cons_h(res, θ, p)
            end, nothing,
        )
    else
        (nothing, nothing)
    end

    lag_hess_prototype = f.lag_hess_prototype

    lag_h! = if f.cons !== nothing && lag_h == true && f.lag_h === nothing
        _lag_prep = prepare_hessian(
            lagrangian, soadtype, x, Constant(one(eltype(x))),
            Constant(ones(eltype(x), num_cons)), Constant(p)
        )
        lag_hess_prototype = zeros(Bool, length(x), length(x))

        if p !== SciMLBase.NullParameters() && p !== nothing
            let lagrangian = lagrangian, _lag_prep = _lag_prep, soadtype = soadtype, cons_h_weighted! = cons_h_weighted!, cons_h! = cons_h!
                function _lag_h!(H::AbstractMatrix, θ, σ, λ, p = p)
                    return if σ == zero(eltype(θ))
                        # When σ=0, use the weighted sum function
                        cons_h_weighted!(H, θ, λ)
                    else
                        hessian!(
                            lagrangian, H, _lag_prep, soadtype, θ,
                            Constant(σ), Constant(λ), Constant(p)
                        )
                    end
                end
                function _lag_h!(h::AbstractVector, θ, σ, λ, p = p)
                    H = hessian(
                        lagrangian, _lag_prep, soadtype, θ, Constant(σ), Constant(λ), Constant(p)
                    )
                    k = 0
                    for i in 1:length(θ)
                        for j in 1:i
                            k += 1
                            h[k] = H[i, j]
                        end
                    end
                    return
                end
                _lag_h!
            end
        else
            let lagrangian = lagrangian, _lag_prep = _lag_prep, soadtype = soadtype, cons_h_weighted! = cons_h_weighted!, p = p
                function _lag_h!(H::AbstractMatrix, θ, σ, λ, p = p)
                    return if σ == zero(eltype(θ))
                        # When σ=0, use the weighted sum function
                        cons_h_weighted!(H, θ, λ)
                    else
                        hessian!(
                            lagrangian, H, _lag_prep, soadtype, θ,
                            Constant(σ), Constant(λ), Constant(p)
                        )
                    end
                end
                function _lag_h!(h::AbstractVector, θ, σ, λ, p = p)
                    H = hessian(
                        lagrangian, _lag_prep, soadtype, θ, Constant(σ), Constant(λ), Constant(p)
                    )
                    k = 0
                    for i in 1:length(θ)
                        for j in 1:i
                            k += 1
                            h[k] = H[i, j]
                        end
                    end
                    return
                end
                _lag_h!
            end
        end
    elseif lag_h == true && f.cons !== nothing
        let f = f, p = p
            (res, θ, σ, μ, p = p) -> f.lag_h(res, θ, σ, μ, p)
        end
    else
        nothing
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

function instantiate_function(
        f::OptimizationFunction{true}, cache::OptimizationBase.ReInitCache,
        adtype::ADTypes.AbstractADType, num_cons = 0;
        kwargs...
    )
    x = cache.u0
    p = cache.p

    return instantiate_function(f, x, adtype, p, num_cons; kwargs...)
end

function instantiate_function(
        f::OptimizationFunction{false}, x, adtype::ADTypes.AbstractADType,
        p = SciMLBase.NullParameters(), num_cons = 0;
        g = false, h = false, hv = false, fg = false, fgh = false,
        cons_j = false, cons_vjp = false, cons_jvp = false, cons_h = false,
        lag_h = false
    )
    adtype, soadtype = generate_adtype(adtype)

    # Create gradient closures with proper type stability using let blocks
    grad = if g == true && f.grad === nothing
        _prep_grad = prepare_gradient(f.f, adtype, x, Constant(p))
        if p !== SciMLBase.NullParameters() && p !== nothing
            let _prep_grad = _prep_grad, f = f, adtype = adtype
                (θ, p = p) -> gradient(f.f, _prep_grad, adtype, θ, Constant(p))
            end
        else
            let _prep_grad = _prep_grad, f = f, adtype = adtype, p = p
                (θ, p = p) -> gradient(f.f, _prep_grad, adtype, θ, Constant(p))
            end
        end
    elseif g == true
        (θ, p = p) -> f.grad(θ, p)
    else
        nothing
    end

    # Create fg! closures
    fg! = if fg == true && f.fg === nothing
        _prep_grad_fg = prepare_gradient(f.f, adtype, x, Constant(p))
        if p !== SciMLBase.NullParameters() && p !== nothing
            let _prep_grad_fg = _prep_grad_fg, f = f, adtype = adtype
                function (θ, p = p)
                    (y, res) = value_and_gradient(f.f, _prep_grad_fg, adtype, θ, Constant(p))
                    return y, res
                end
            end
        else
            let _prep_grad_fg = _prep_grad_fg, f = f, adtype = adtype, p = p
                function (θ, p = p)
                    (y, res) = value_and_gradient(f.f, _prep_grad_fg, adtype, θ, Constant(p))
                    return y, res
                end
            end
        end
    elseif fg == true
        (θ, p = p) -> f.fg(θ, p)
    else
        nothing
    end

    hess_sparsity = f.hess_prototype
    hess_colors = f.hess_colorvec

    # Create hessian closures
    _prep_hess = if (h == true && f.hess === nothing) || (fgh == true && f.fgh === nothing)
        prepare_hessian(f.f, soadtype, x, Constant(p))
    else
        nothing
    end

    hess = if h == true && f.hess === nothing
        if p !== SciMLBase.NullParameters() && p !== nothing
            let _prep_hess = _prep_hess, f = f, soadtype = soadtype
                (θ, p = p) -> hessian(f.f, _prep_hess, soadtype, θ, Constant(p))
            end
        else
            let _prep_hess = _prep_hess, f = f, soadtype = soadtype, p = p
                (θ, p = p) -> hessian(f.f, _prep_hess, soadtype, θ, Constant(p))
            end
        end
    elseif h == true
        (θ, p = p) -> f.hess(θ, p)
    else
        nothing
    end

    fgh! = if fgh == true && f.fgh === nothing
        if p !== SciMLBase.NullParameters() && p !== nothing
            let _prep_hess = _prep_hess, f = f, adtype = adtype
                function (θ, p = p)
                    (y, G, H) = value_derivative_and_second_derivative(
                        f.f, _prep_hess, adtype, θ, Constant(p)
                    )
                    return y, G, H
                end
            end
        else
            let _prep_hess = _prep_hess, f = f, adtype = adtype, p = p
                function (θ, p = p)
                    (y, G, H) = value_derivative_and_second_derivative(
                        f.f, _prep_hess, adtype, θ, Constant(p)
                    )
                    return y, G, H
                end
            end
        end
    elseif fgh == true
        (θ, p = p) -> f.fgh(θ, p)
    else
        nothing
    end

    hv! = if hv == true && f.hv === nothing
        _prep_hvp = prepare_hvp(f.f, soadtype, x, (zeros(eltype(x), size(x)),), Constant(p))
        if p !== SciMLBase.NullParameters() && p !== nothing
            let _prep_hvp = _prep_hvp, f = f, soadtype = soadtype
                (θ, v, p = p) -> only(hvp(f.f, _prep_hvp, soadtype, θ, (v,), Constant(p)))
            end
        else
            let _prep_hvp = _prep_hvp, f = f, soadtype = soadtype, p = p
                (θ, v, p = p) -> only(hvp(f.f, _prep_hvp, soadtype, θ, (v,), Constant(p)))
            end
        end
    elseif hv == true
        (θ, v, p = p) -> f.hv(θ, v, p)
    else
        nothing
    end

    # Create constraint-related closures
    cons = if f.cons !== nothing
        Base.Fix2(f.cons, p)
    else
        nothing
    end

    lagrangian = if f.cons !== nothing
        let f = f
            (θ, σ, λ, p) -> σ * f.f(θ, p) + dot(λ, f.cons(θ, p))
        end
    else
        nothing
    end

    cons_jac_prototype = f.cons_jac_prototype
    cons_jac_colorvec = f.cons_jac_colorvec

    cons_j! = if f.cons !== nothing && cons_j == true && f.cons_j === nothing
        _prep_jac = prepare_jacobian(f.cons, adtype, x, Constant(p))
        let f = f, _prep_jac = _prep_jac, adtype = adtype, p = p
            function (θ)
                J = jacobian(f.cons, _prep_jac, adtype, θ, Constant(p))
                if size(J, 1) == 1
                    J = vec(J)
                end
                return J
            end
        end
    elseif cons_j == true && f.cons !== nothing
        let f = f, p = p
            (θ) -> f.cons_j(θ, p)
        end
    else
        nothing
    end

    cons_vjp! = if f.cons_vjp === nothing && cons_vjp == true && f.cons !== nothing
        _prep_pullback = prepare_pullback(
            f.cons, adtype, x, (ones(eltype(x), num_cons),), Constant(p)
        )
        let f = f, _prep_pullback = _prep_pullback, adtype = adtype, p = p
            (θ, v) -> only(pullback(f.cons, _prep_pullback, adtype, θ, (v,), Constant(p)))
        end
    elseif cons_vjp == true && f.cons !== nothing
        let f = f, p = p
            (θ, v) -> f.cons_vjp(θ, v, p)
        end
    else
        nothing
    end

    cons_jvp! = if f.cons_jvp === nothing && cons_jvp == true && f.cons !== nothing
        _prep_pushforward = prepare_pushforward(
            f.cons, adtype, x, (ones(eltype(x), length(x)),), Constant(p)
        )
        let f = f, _prep_pushforward = _prep_pushforward, adtype = adtype, p = p
            (θ, v) -> only(pushforward(f.cons, _prep_pushforward, adtype, θ, (v,), Constant(p)))
        end
    elseif cons_jvp == true && f.cons !== nothing
        let f = f, p = p
            (θ, v) -> f.cons_jvp(θ, v, p)
        end
    else
        nothing
    end

    conshess_sparsity = f.cons_hess_prototype
    conshess_colors = f.cons_hess_colorvec

    cons_h! = if f.cons !== nothing && cons_h == true && f.cons_h === nothing
        _cons_i = let f = f, p = p
            (x, i) -> f.cons(x, p)[i]
        end
        _prep_cons_hess = [
            prepare_hessian(_cons_i, soadtype, x, Constant(i))
                for i in 1:num_cons
        ]
        let _cons_i = _cons_i, _prep_cons_hess = _prep_cons_hess, soadtype = soadtype, num_cons = num_cons
            function (θ)
                H = map(1:num_cons) do i
                    hessian(_cons_i, _prep_cons_hess[i], soadtype, θ, Constant(i))
                end
                return H
            end
        end
    elseif cons_h == true && f.cons !== nothing
        let f = f, p = p
            (θ) -> f.cons_h(θ, p)
        end
    else
        nothing
    end

    lag_hess_prototype = f.lag_hess_prototype

    lag_h! = if f.cons !== nothing && lag_h == true && f.lag_h === nothing
        _lag_prep = prepare_hessian(
            lagrangian, soadtype, x, Constant(one(eltype(x))),
            Constant(ones(eltype(x), num_cons)), Constant(p)
        )
        lag_hess_prototype = zeros(Bool, length(x), length(x))

        if p !== SciMLBase.NullParameters() && p !== nothing
            let lagrangian = lagrangian, _lag_prep = _lag_prep, soadtype = soadtype, cons_h! = cons_h!
                function _lag_h!(θ, σ, λ, p = p)
                    if σ == zero(eltype(θ))
                        return λ .* cons_h!(θ)
                    else
                        return hessian(
                            lagrangian, _lag_prep, soadtype, θ,
                            Constant(σ), Constant(λ), Constant(p)
                        )
                    end
                end
                _lag_h!
            end
        else
            let lagrangian = lagrangian, _lag_prep = _lag_prep, soadtype = soadtype, cons_h! = cons_h!, p = p
                function _lag_h!(θ, σ, λ, p = p)
                    if σ == zero(eltype(θ))
                        return λ .* cons_h!(θ)
                    else
                        return hessian(
                            lagrangian, _lag_prep, soadtype, θ,
                            Constant(σ), Constant(λ), Constant(p)
                        )
                    end
                end
                _lag_h!
            end
        end
    elseif lag_h == true && f.cons !== nothing
        let f = f, p = p
            (θ, σ, λ, p = p) -> f.lag_h(θ, σ, λ, p)
        end
    else
        nothing
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
        sys = f.sys,
        expr = f.expr,
        cons_expr = f.cons_expr
    )
end

function instantiate_function(
        f::OptimizationFunction{false}, cache::OptimizationBase.ReInitCache,
        adtype::ADTypes.AbstractADType, num_cons = 0; kwargs...
    )
    x = cache.u0
    p = cache.p

    return instantiate_function(f, x, adtype, p, num_cons; kwargs...)
end
