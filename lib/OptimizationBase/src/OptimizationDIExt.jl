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
    value_gradient_and_hessian!, value_gradient_and_hessian,
    gradient!, hessian!, hvp!, jacobian!, gradient, hessian,
    hvp, jacobian, Constant
using ADTypes, SciMLBase

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

    if g == true && f.grad === nothing
        prep_grad = prepare_gradient(f.f, adtype, x, Constant(p))
        # Use let block to capture prep_grad with concrete type and avoid Core.Box
        grad = let _prep_grad = prep_grad, _f = f.f, _adtype = adtype, _p = p
            if _p !== SciMLBase.NullParameters() && _p !== nothing
                (res, θ, p = _p) -> gradient!(_f, res, _prep_grad, _adtype, θ, Constant(p))
            else
                (res, θ) -> gradient!(_f, res, _prep_grad, _adtype, θ, Constant(_p))
            end
        end
    elseif g == true
        grad = (G, θ, p = p) -> f.grad(G, θ, p)
    else
        grad = nothing
    end

    if fg == true && f.fg === nothing
        if g == false
            prep_grad = prepare_gradient(f.f, adtype, x, Constant(p))
        end
        # Use let block to capture prep_grad with concrete type and avoid Core.Box
        fg! = let _prep_grad = prep_grad, _f = f.f, _adtype = adtype, _p = p
            if _p !== SciMLBase.NullParameters() && _p !== nothing
                function (res, θ, p = _p)
                    (y, _) = value_and_gradient!(_f, res, _prep_grad, _adtype, θ, Constant(p))
                    return y
                end
            else
                function (res, θ)
                    (y, _) = value_and_gradient!(_f, res, _prep_grad, _adtype, θ, Constant(_p))
                    return y
                end
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
        prep_hess = prepare_hessian(f.f, soadtype, x, Constant(p))
        # Use let block to capture prep_hess with concrete type and avoid Core.Box
        hess = let _prep_hess = prep_hess, _f = f.f, _soadtype = soadtype, _p = p
            if _p !== SciMLBase.NullParameters() && _p !== nothing
                (res, θ, p = _p) -> hessian!(_f, res, _prep_hess, _soadtype, θ, Constant(p))
            else
                (res, θ) -> hessian!(_f, res, _prep_hess, _soadtype, θ, Constant(_p))
            end
        end
    elseif h == true
        hess = (H, θ, p = p) -> f.hess(H, θ, p)
    else
        hess = nothing
    end

    if fgh == true && f.fgh === nothing
        # Use let block to capture prep_hess with concrete type and avoid Core.Box
        fgh! = let _prep_hess = prep_hess, _f = f.f, _soadtype = soadtype, _p = p
            if _p !== SciMLBase.NullParameters() && _p !== nothing
                function (G, H, θ, p = _p)
                    (y, _, _) = value_gradient_and_hessian!(
                        _f, G, H, _prep_hess, _soadtype, θ, Constant(p)
                    )
                    return y
                end
            else
                function (G, H, θ)
                    (y, _, _) = value_gradient_and_hessian!(
                        _f, G, H, _prep_hess, _soadtype, θ, Constant(_p)
                    )
                    return y
                end
            end
        end
    elseif fgh == true
        fgh! = (G, H, θ, p = p) -> f.fgh(G, H, θ, p)
    else
        fgh! = nothing
    end

    if hv == true && f.hv === nothing
        prep_hvp = prepare_hvp(f.f, soadtype, x, (zeros(eltype(x), size(x)),), Constant(p))
        # Use let block to capture prep_hvp with concrete type and avoid Core.Box
        hv! = let _prep_hvp = prep_hvp, _f = f.f, _soadtype = soadtype, _p = p
            if _p !== SciMLBase.NullParameters() && _p !== nothing
                function (H, θ, v, p = _p)
                    return only(hvp!(_f, (H,), _prep_hvp, _soadtype, θ, (v,), Constant(p)))
                end
            else
                function (H, θ, v)
                    return only(hvp!(_f, (H,), _prep_hvp, _soadtype, θ, (v,), Constant(_p)))
                end
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
        cons = (res, x) -> f.cons(res, x, p)
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
            return σ * f.f(θ, p) + dot(λ, cons_oop(θ))
        end
    end

    cons_jac_prototype = f.cons_jac_prototype
    cons_jac_colorvec = f.cons_jac_colorvec
    if f.cons !== nothing && cons_j == true && f.cons_j === nothing
        prep_jac = prepare_jacobian(cons_oop, adtype, x)
        # Use let block to capture prep_jac with concrete type and avoid Core.Box
        cons_j! = let _prep_jac = prep_jac, _cons_oop = cons_oop, _adtype = adtype
            function (J, θ)
                jacobian!(_cons_oop, J, _prep_jac, _adtype, θ)
                return if size(J, 1) == 1
                    J = vec(J)
                end
            end
        end
    elseif cons_j == true && f.cons !== nothing
        cons_j! = (J, θ) -> f.cons_j(J, θ, p)
    else
        cons_j! = nothing
    end

    if f.cons_vjp === nothing && cons_vjp == true && f.cons !== nothing
        prep_pullback = prepare_pullback(cons_oop, adtype, x, (ones(eltype(x), num_cons),))
        # Use let block to capture prep_pullback with concrete type and avoid Core.Box
        cons_vjp! = let _prep_pullback = prep_pullback, _cons_oop = cons_oop, _adtype = adtype
            (J, θ, v) -> only(pullback!(_cons_oop, (J,), _prep_pullback, _adtype, θ, (v,)))
        end
    elseif cons_vjp == true && f.cons !== nothing
        cons_vjp! = (J, θ, v) -> f.cons_vjp(J, θ, v, p)
    else
        cons_vjp! = nothing
    end

    if f.cons_jvp === nothing && cons_jvp == true && f.cons !== nothing
        prep_pushforward = prepare_pushforward(
            cons_oop, adtype, x, (ones(eltype(x), length(x)),)
        )
        # Use let block to capture prep_pushforward with concrete type and avoid Core.Box
        cons_jvp! = let _prep_pushforward = prep_pushforward, _cons_oop = cons_oop, _adtype = adtype
            (J, θ, v) -> only(pushforward!(_cons_oop, (J,), _prep_pushforward, _adtype, θ, (v,)))
        end
    elseif cons_jvp == true && f.cons !== nothing
        cons_jvp! = (J, θ, v) -> f.cons_jvp(J, θ, v, p)
    else
        cons_jvp! = nothing
    end

    conshess_sparsity = f.cons_hess_prototype
    conshess_colors = f.cons_hess_colorvec

    # Prepare constraint Hessian preparations if needed by lag_h or cons_h
    if f.cons !== nothing && f.cons_h === nothing && (cons_h == true || lag_h == true)
        # This is necessary because DI will create a symbolic index for `Constant(i)`
        # to trace into the function, since it assumes `Constant` can change between
        # DI calls.
        if adtype isa ADTypes.AutoSymbolics
            prep_cons_hess = [
                prepare_hessian(Base.Fix2(cons_oop, i), soadtype, x)
                    for i in 1:num_cons
            ]
        else
            prep_cons_hess = [
                prepare_hessian(cons_oop, soadtype, x, Constant(i))
                    for i in 1:num_cons
            ]
        end
    else
        prep_cons_hess = nothing
    end

    # Generate cons_h! functions
    if f.cons !== nothing && f.cons_h === nothing && prep_cons_hess !== nothing
        # Standard cons_h! that returns array of matrices
        if cons_h == true
            # Use let block to capture prep_cons_hess with concrete type and avoid Core.Box
            if adtype isa ADTypes.AutoSymbolics
                cons_h! = let _prep_cons_hess = prep_cons_hess, _cons_oop = cons_oop, _soadtype = soadtype, _num_cons = num_cons
                    function (H, θ)
                        for i in 1:_num_cons
                            hessian!(Base.Fix2(_cons_oop, i), H[i], _prep_cons_hess[i], _soadtype, θ)
                        end
                        return
                    end
                end
            else
                cons_h! = let _prep_cons_hess = prep_cons_hess, _cons_oop = cons_oop, _soadtype = soadtype, _num_cons = num_cons
                    function (H, θ)
                        for i in 1:_num_cons
                            hessian!(_cons_oop, H[i], _prep_cons_hess[i], _soadtype, θ, Constant(i))
                        end
                        return
                    end
                end
            end
        else
            cons_h! = nothing
        end

        # Weighted sum dispatch for cons_h! (always created if prep_cons_hess exists)
        # This is used by lag_h! when σ=0
        # Use let block to capture prep_cons_hess with concrete type and avoid Core.Box
        cons_h_weighted! = let _prep_cons_hess = prep_cons_hess, _cons_oop = cons_oop, _soadtype = soadtype, _num_cons = num_cons
            function (H::AbstractMatrix, θ, λ)
                # Compute weighted sum: H = Σᵢ λᵢ∇²cᵢ
                H .= zero(eltype(H))

                # Create a single temporary matrix to reuse for all constraints
                Hi = similar(H)

                for i in 1:_num_cons
                    if λ[i] != zero(eltype(λ))
                        # Compute constraint's Hessian into temporary matrix
                        hessian!(_cons_oop, Hi, _prep_cons_hess[i], _soadtype, θ, Constant(i))
                        # Add weighted Hessian to result using in-place operation
                        # H += λ[i] * Hi
                        @. H += λ[i] * Hi
                    end
                end
                return
            end
        end
    elseif cons_h == true && f.cons !== nothing
        cons_h! = (res, θ) -> f.cons_h(res, θ, p)
        cons_h_weighted! = nothing
    else
        cons_h! = nothing
        cons_h_weighted! = nothing
    end

    lag_hess_prototype = f.lag_hess_prototype

    if f.cons !== nothing && lag_h == true && f.lag_h === nothing
        lag_prep = prepare_hessian(
            lagrangian, soadtype, x, Constant(one(eltype(x))),
            Constant(ones(eltype(x), num_cons)), Constant(p)
        )
        lag_hess_prototype = zeros(Bool, length(x), length(x))

        # Use let block to capture lag_prep and cons_h_weighted! with concrete types
        lag_h! = let _lag_prep = lag_prep, _lagrangian = lagrangian, _soadtype = soadtype, _p = p, _cons_h_weighted = cons_h_weighted!, _cons_h = cons_h!
            if _p !== SciMLBase.NullParameters() && _p !== nothing
                # Version with parameter p
                function (H_or_h, θ, σ, λ, p = _p)
                    if H_or_h isa AbstractMatrix
                        return if σ == zero(eltype(θ))
                            _cons_h_weighted(H_or_h, θ, λ)
                        else
                            hessian!(
                                _lagrangian, H_or_h, _lag_prep, _soadtype, θ,
                                Constant(σ), Constant(λ), Constant(p)
                            )
                        end
                    else
                        H = hessian(
                            _lagrangian, _lag_prep, _soadtype, θ,
                            Constant(σ), Constant(λ), Constant(p)
                        )
                        k = 0
                        for i in 1:length(θ)
                            for j in 1:i
                                k += 1
                                H_or_h[k] = H[i, j]
                            end
                        end
                        return
                    end
                end
            else
                # Version without parameter p
                function (H_or_h, θ, σ, λ)
                    if H_or_h isa AbstractMatrix
                        return if σ == zero(eltype(θ))
                            _cons_h_weighted(H_or_h, θ, λ)
                        else
                            hessian!(
                                _lagrangian, H_or_h, _lag_prep, _soadtype, θ,
                                Constant(σ), Constant(λ), Constant(_p)
                            )
                        end
                    else
                        H = hessian(
                            _lagrangian, _lag_prep, _soadtype, θ,
                            Constant(σ), Constant(λ), Constant(_p)
                        )
                        k = 0
                        for i in 1:length(θ)
                            for j in 1:i
                                k += 1
                                H_or_h[k] = H[i, j]
                            end
                        end
                        return
                    end
                end
            end
        end
    elseif lag_h == true && f.cons !== nothing
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

    if g == true && f.grad === nothing
        prep_grad = prepare_gradient(f.f, adtype, x, Constant(p))
        # Use let block to capture prep_grad with concrete type and avoid Core.Box
        grad = let _prep_grad = prep_grad, _f = f.f, _adtype = adtype, _p = p
            if _p !== SciMLBase.NullParameters() && _p !== nothing
                (θ, p = _p) -> gradient(_f, _prep_grad, _adtype, θ, Constant(p))
            else
                (θ) -> gradient(_f, _prep_grad, _adtype, θ, Constant(_p))
            end
        end
    elseif g == true
        grad = (θ, p = p) -> f.grad(θ, p)
    else
        grad = nothing
    end

    if fg == true && f.fg === nothing
        if g == false
            prep_grad = prepare_gradient(f.f, adtype, x, Constant(p))
        end
        # Use let block to capture prep_grad with concrete type and avoid Core.Box
        fg! = let _prep_grad = prep_grad, _f = f.f, _adtype = adtype, _p = p
            if _p !== SciMLBase.NullParameters() && _p !== nothing
                function (θ, p = _p)
                    (y, res) = value_and_gradient(_f, _prep_grad, _adtype, θ, Constant(p))
                    return y, res
                end
            else
                function (θ)
                    (y, res) = value_and_gradient(_f, _prep_grad, _adtype, θ, Constant(_p))
                    return y, res
                end
            end
        end
    elseif fg == true
        fg! = (θ, p = p) -> f.fg(θ, p)
    else
        fg! = nothing
    end

    hess_sparsity = f.hess_prototype
    hess_colors = f.hess_colorvec
    if h == true && f.hess === nothing
        prep_hess = prepare_hessian(f.f, soadtype, x, Constant(p))
        # Use let block to capture prep_hess with concrete type and avoid Core.Box
        hess = let _prep_hess = prep_hess, _f = f.f, _soadtype = soadtype, _p = p
            if _p !== SciMLBase.NullParameters() && _p !== nothing
                (θ, p = _p) -> hessian(_f, _prep_hess, _soadtype, θ, Constant(p))
            else
                (θ) -> hessian(_f, _prep_hess, _soadtype, θ, Constant(_p))
            end
        end
    elseif h == true
        hess = (θ, p = p) -> f.hess(θ, p)
    else
        hess = nothing
    end

    if fgh == true && f.fgh === nothing
        # Use let block to capture prep_hess with concrete type and avoid Core.Box
        fgh! = let _prep_hess = prep_hess, _f = f.f, _soadtype = soadtype, _p = p
            if _p !== SciMLBase.NullParameters() && _p !== nothing
                function (θ, p = _p)
                    (y, G, H) = value_gradient_and_hessian(
                        _f, _prep_hess, _soadtype, θ, Constant(p)
                    )
                    return y, G, H
                end
            else
                function (θ)
                    (y, G, H) = value_gradient_and_hessian(
                        _f, _prep_hess, _soadtype, θ, Constant(_p)
                    )
                    return y, G, H
                end
            end
        end
    elseif fgh == true
        fgh! = (θ, p = p) -> f.fgh(θ, p)
    else
        fgh! = nothing
    end

    if hv == true && f.hv === nothing
        prep_hvp = prepare_hvp(f.f, soadtype, x, (zeros(eltype(x), size(x)),), Constant(p))
        # Use let block to capture prep_hvp with concrete type and avoid Core.Box
        hv! = let _prep_hvp = prep_hvp, _f = f.f, _soadtype = soadtype, _p = p
            if _p !== SciMLBase.NullParameters() && _p !== nothing
                (θ, v, p = _p) -> only(hvp(_f, _prep_hvp, _soadtype, θ, (v,), Constant(p)))
            else
                (θ, v) -> only(hvp(_f, _prep_hvp, _soadtype, θ, (v,), Constant(_p)))
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
        cons = Base.Fix2(f.cons, p)

        function lagrangian(θ, σ, λ, p)
            return σ * f.f(θ, p) + dot(λ, f.cons(θ, p))
        end
    end

    cons_jac_prototype = f.cons_jac_prototype
    cons_jac_colorvec = f.cons_jac_colorvec
    if f.cons !== nothing && cons_j == true && f.cons_j === nothing
        prep_jac = prepare_jacobian(f.cons, adtype, x, Constant(p))
        # Use let block to capture prep_jac with concrete type and avoid Core.Box
        cons_j! = let _prep_jac = prep_jac, _cons = f.cons, _adtype = adtype, _p = p
            function (θ)
                J = jacobian(_cons, _prep_jac, _adtype, θ, Constant(_p))
                if size(J, 1) == 1
                    J = vec(J)
                end
                return J
            end
        end
    elseif cons_j == true && f.cons !== nothing
        cons_j! = (θ) -> f.cons_j(θ, p)
    else
        cons_j! = nothing
    end

    if f.cons_vjp === nothing && cons_vjp == true && f.cons !== nothing
        prep_pullback = prepare_pullback(
            f.cons, adtype, x, (ones(eltype(x), num_cons),), Constant(p)
        )
        # Use let block to capture prep_pullback with concrete type and avoid Core.Box
        cons_vjp! = let _prep_pullback = prep_pullback, _cons = f.cons, _adtype = adtype, _p = p
            (θ, v) -> only(pullback(_cons, _prep_pullback, _adtype, θ, (v,), Constant(_p)))
        end
    elseif cons_vjp == true && f.cons !== nothing
        cons_vjp! = (θ, v) -> f.cons_vjp(θ, v, p)
    else
        cons_vjp! = nothing
    end

    if f.cons_jvp === nothing && cons_jvp == true && f.cons !== nothing
        prep_pushforward = prepare_pushforward(
            f.cons, adtype, x, (ones(eltype(x), length(x)),), Constant(p)
        )
        # Use let block to capture prep_pushforward with concrete type and avoid Core.Box
        cons_jvp! = let _prep_pushforward = prep_pushforward, _cons = f.cons, _adtype = adtype, _p = p
            (θ, v) -> only(pushforward(_cons, _prep_pushforward, _adtype, θ, (v,), Constant(_p)))
        end
    elseif cons_jvp == true && f.cons !== nothing
        cons_jvp! = (θ, v) -> f.cons_jvp(θ, v, p)
    else
        cons_jvp! = nothing
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

        # Use let block to capture prep_cons_hess with concrete type and avoid Core.Box
        cons_h! = let _prep_cons_hess = prep_cons_hess, _cons_i = cons_i, _soadtype = soadtype, _num_cons = num_cons
            function (θ)
                H = map(1:_num_cons) do i
                    hessian(_cons_i, _prep_cons_hess[i], _soadtype, θ, Constant(i))
                end
                return H
            end
        end
    elseif cons_h == true && f.cons !== nothing
        cons_h! = (θ) -> f.cons_h(θ, p)
    else
        cons_h! = nothing
    end

    lag_hess_prototype = f.lag_hess_prototype

    if f.cons !== nothing && lag_h == true && f.lag_h === nothing
        lag_prep = prepare_hessian(
            lagrangian, soadtype, x, Constant(one(eltype(x))),
            Constant(ones(eltype(x), num_cons)), Constant(p)
        )
        lag_hess_prototype = zeros(Bool, length(x), length(x))

        # Use let block to capture lag_prep with concrete type and avoid Core.Box
        lag_h! = let _lag_prep = lag_prep, _lagrangian = lagrangian, _soadtype = soadtype, _p = p, _cons_h = cons_h!
            if _p !== SciMLBase.NullParameters() && _p !== nothing
                function (θ, σ, λ, p = _p)
                    if σ == zero(eltype(θ))
                        return λ .* _cons_h(θ)
                    else
                        return hessian(
                            _lagrangian, _lag_prep, _soadtype, θ,
                            Constant(σ), Constant(λ), Constant(p)
                        )
                    end
                end
            else
                function (θ, σ, λ)
                    if σ == zero(eltype(θ))
                        return λ .* _cons_h(θ)
                    else
                        return hessian(
                            _lagrangian, _lag_prep, _soadtype, θ,
                            Constant(σ), Constant(λ), Constant(_p)
                        )
                    end
                end
            end
        end
    elseif lag_h == true && f.cons !== nothing
        lag_h! = (θ, σ, λ, p = p) -> f.lag_h(θ, σ, λ, p)
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
