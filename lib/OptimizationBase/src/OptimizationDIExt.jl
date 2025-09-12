using OptimizationBase
import OptimizationBase.ArrayInterface
import OptimizationBase.SciMLBase: OptimizationFunction
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

function instantiate_function(
        f::OptimizationFunction{true}, x, adtype::ADTypes.AbstractADType,
        p = SciMLBase.NullParameters(), num_cons = 0;
        g = false, h = false, hv = false, fg = false, fgh = false,
        cons_j = false, cons_vjp = false, cons_jvp = false, cons_h = false,
        lag_h = false)
    adtype, soadtype = generate_adtype(adtype)

    if g == true && f.grad === nothing
        prep_grad = prepare_gradient(f.f, adtype, x, Constant(p))
        function grad(res, θ)
            gradient!(f.f, res, prep_grad, adtype, θ, Constant(p))
        end
        if p !== SciMLBase.NullParameters() && p !== nothing
            function grad(res, θ, p)
                gradient!(f.f, res, prep_grad, adtype, θ, Constant(p))
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
        prep_hess = prepare_hessian(f.f, soadtype, x, Constant(p))
        function hess(res, θ)
            hessian!(f.f, res, prep_hess, soadtype, θ, Constant(p))
        end
        if p !== SciMLBase.NullParameters() && p !== nothing
            function hess(res, θ, p)
                hessian!(f.f, res, prep_hess, soadtype, θ, Constant(p))
            end
        end
    elseif h == true
        hess = (H, θ, p = p) -> f.hess(H, θ, p)
    else
        hess = nothing
    end

    if fgh == true && f.fgh === nothing
        function fgh!(G, H, θ)
            (y,
                _,
                _) = value_derivative_and_second_derivative!(
                f.f, G, H, prep_hess, soadtype, θ, Constant(p))
            return y
        end
        if p !== SciMLBase.NullParameters() && p !== nothing
            function fgh!(G, H, θ, p)
                (y,
                    _,
                    _) = value_derivative_and_second_derivative!(
                    f.f, G, H, prep_hess, soadtype, θ, Constant(p))
                return y
            end
        end
    elseif fgh == true
        fgh! = (G, H, θ, p = p) -> f.fgh(G, H, θ, p)
    else
        fgh! = nothing
    end

    if hv == true && f.hv === nothing
        prep_hvp = prepare_hvp(f.f, soadtype, x, (zeros(eltype(x), size(x)),), Constant(p))
        function hv!(H, θ, v)
            only(hvp!(f.f, (H,), prep_hvp, soadtype, θ, (v,), Constant(p)))
        end
        if p !== SciMLBase.NullParameters() && p !== nothing
            function hv!(H, θ, v, p)
                only(hvp!(f.f, (H,), soadtype, θ, (v,), Constant(p)))
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
        function cons_j!(J, θ)
            jacobian!(cons_oop, J, prep_jac, adtype, θ)
            if size(J, 1) == 1
                J = vec(J)
            end
        end
    elseif cons_j == true && f.cons !== nothing
        cons_j! = (J, θ) -> f.cons_j(J, θ, p)
    else
        cons_j! = nothing
    end

    if f.cons_vjp === nothing && cons_vjp == true && f.cons !== nothing
        prep_pullback = prepare_pullback(cons_oop, adtype, x, (ones(eltype(x), num_cons),))
        function cons_vjp!(J, θ, v)
            only(pullback!(cons_oop, (J,), prep_pullback, adtype, θ, (v,)))
        end
    elseif cons_vjp == true && f.cons !== nothing
        cons_vjp! = (J, θ, v) -> f.cons_vjp(J, θ, v, p)
    else
        cons_vjp! = nothing
    end

    if f.cons_jvp === nothing && cons_jvp == true && f.cons !== nothing
        prep_pushforward = prepare_pushforward(
            cons_oop, adtype, x, (ones(eltype(x), length(x)),))
        function cons_jvp!(J, θ, v)
            only(pushforward!(cons_oop, (J,), prep_pushforward, adtype, θ, (v,)))
        end
    elseif cons_jvp == true && f.cons !== nothing
        cons_jvp! = (J, θ, v) -> f.cons_jvp(J, θ, v, p)
    else
        cons_jvp! = nothing
    end

    conshess_sparsity = f.cons_hess_prototype
    conshess_colors = f.cons_hess_colorvec
    if f.cons !== nothing && f.cons_h === nothing && cons_h == true
        prep_cons_hess = [prepare_hessian(cons_oop, soadtype, x, Constant(i))
                          for i in 1:num_cons]

        function cons_h!(H, θ)
            for i in 1:num_cons
                hessian!(cons_oop, H[i], prep_cons_hess[i], soadtype, θ, Constant(i))
            end
        end
    elseif cons_h == true && f.cons !== nothing
        cons_h! = (res, θ) -> f.cons_h(res, θ, p)
    else
        cons_h! = nothing
    end

    lag_hess_prototype = f.lag_hess_prototype

    if f.cons !== nothing && lag_h == true && f.lag_h === nothing
        lag_prep = prepare_hessian(
            lagrangian, soadtype, x, Constant(one(eltype(x))),
            Constant(ones(eltype(x), num_cons)), Constant(p))
        lag_hess_prototype = zeros(Bool, num_cons, length(x))

        function lag_h!(H::AbstractMatrix, θ, σ, λ)
            if σ == zero(eltype(θ))
                cons_h!(H, θ)
                H *= λ
            else
                hessian!(lagrangian, H, lag_prep, soadtype, θ,
                    Constant(σ), Constant(λ), Constant(p))
            end
        end

        function lag_h!(h::AbstractVector, θ, σ, λ)
            H = hessian(
                lagrangian, lag_prep, soadtype, θ, Constant(σ), Constant(λ), Constant(p))
            k = 0
            for i in 1:length(θ)
                for j in 1:i
                    k += 1
                    h[k] = H[i, j]
                end
            end
        end

        if p !== SciMLBase.NullParameters() && p !== nothing
            function lag_h!(H::AbstractMatrix, θ, σ, λ, p)
                if σ == zero(eltype(θ))
                    cons_h!(H, θ)
                    H *= λ
                else
                    hessian!(lagrangian, H, lag_prep, soadtype, θ,
                        Constant(σ), Constant(λ), Constant(p))
                end
            end

            function lag_h!(h::AbstractVector, θ, σ, λ, p)
                H = hessian(lagrangian, lag_prep, soadtype, θ,
                    Constant(σ), Constant(λ), Constant(p))
                k = 0
                for i in 1:length(θ)
                    for j in 1:i
                        k += 1
                        h[k] = H[i, j]
                    end
                end
            end
        end
    elseif lag_h == true && f.cons !== nothing
        lag_h! = (res, θ, σ, μ, p = p) -> f.lag_h(res, θ, σ, μ, p)
    else
        lag_h! = nothing
    end

    return OptimizationFunction{true}(f.f, adtype;
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
        cons_expr = f.cons_expr)
end

function instantiate_function(
        f::OptimizationFunction{true}, cache::OptimizationBase.ReInitCache,
        adtype::ADTypes.AbstractADType, num_cons = 0;
        kwargs...)
    x = cache.u0
    p = cache.p

    return instantiate_function(f, x, adtype, p, num_cons; kwargs...)
end

function instantiate_function(
        f::OptimizationFunction{false}, x, adtype::ADTypes.AbstractADType,
        p = SciMLBase.NullParameters(), num_cons = 0;
        g = false, h = false, hv = false, fg = false, fgh = false,
        cons_j = false, cons_vjp = false, cons_jvp = false, cons_h = false,
        lag_h = false)
    adtype, soadtype = generate_adtype(adtype)

    if g == true && f.grad === nothing
        prep_grad = prepare_gradient(f.f, adtype, x, Constant(p))
        function grad(θ)
            gradient(f.f, prep_grad, adtype, θ, Constant(p))
        end
        if p !== SciMLBase.NullParameters() && p !== nothing
            function grad(θ, p)
                gradient(f.f, prep_grad, adtype, θ, Constant(p))
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
        function fg!(θ)
            (y, res) = value_and_gradient(f.f, prep_grad, adtype, θ, Constant(p))
            return y, res
        end
        if p !== SciMLBase.NullParameters() && p !== nothing
            function fg!(θ, p)
                (y, res) = value_and_gradient(f.f, prep_grad, adtype, θ, Constant(p))
                return y, res
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
        function hess(θ)
            hessian(f.f, prep_hess, soadtype, θ, Constant(p))
        end
        if p !== SciMLBase.NullParameters() && p !== nothing
            function hess(θ, p)
                hessian(f.f, prep_hess, soadtype, θ, Constant(p))
            end
        end
    elseif h == true
        hess = (θ, p = p) -> f.hess(θ, p)
    else
        hess = nothing
    end

    if fgh == true && f.fgh === nothing
        function fgh!(θ)
            (y,
                G,
                H) = value_derivative_and_second_derivative(
                f.f, prep_hess, adtype, θ, Constant(p))
            return y, G, H
        end
        if p !== SciMLBase.NullParameters() && p !== nothing
            function fgh!(θ, p)
                (y,
                    G,
                    H) = value_derivative_and_second_derivative(
                    f.f, prep_hess, adtype, θ, Constant(p))
                return y, G, H
            end
        end
    elseif fgh == true
        fgh! = (θ, p = p) -> f.fgh(θ, p)
    else
        fgh! = nothing
    end

    if hv == true && f.hv === nothing
        prep_hvp = prepare_hvp(f.f, soadtype, x, (zeros(eltype(x), size(x)),), Constant(p))
        function hv!(θ, v)
            only(hvp(f.f, prep_hvp, soadtype, θ, (v,), Constant(p)))
        end
        if p !== SciMLBase.NullParameters() && p !== nothing
            function hv!(θ, v, p)
                only(hvp(f.f, prep_hvp, soadtype, θ, (v,), Constant(p)))
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
        function cons_j!(θ)
            J = jacobian(f.cons, prep_jac, adtype, θ, Constant(p))
            if size(J, 1) == 1
                J = vec(J)
            end
            return J
        end
    elseif cons_j == true && f.cons !== nothing
        cons_j! = (θ) -> f.cons_j(θ, p)
    else
        cons_j! = nothing
    end

    if f.cons_vjp === nothing && cons_vjp == true && f.cons !== nothing
        prep_pullback = prepare_pullback(
            f.cons, adtype, x, (ones(eltype(x), num_cons),), Constant(p))
        function cons_vjp!(θ, v)
            return only(pullback(f.cons, prep_pullback, adtype, θ, (v,), Constant(p)))
        end
    elseif cons_vjp == true && f.cons !== nothing
        cons_vjp! = (θ, v) -> f.cons_vjp(θ, v, p)
    else
        cons_vjp! = nothing
    end

    if f.cons_jvp === nothing && cons_jvp == true && f.cons !== nothing
        prep_pushforward = prepare_pushforward(
            f.cons, adtype, x, (ones(eltype(x), length(x)),), Constant(p))
        function cons_jvp!(θ, v)
            return only(pushforward(f.cons, prep_pushforward, adtype, θ, (v,), Constant(p)))
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
        prep_cons_hess = [prepare_hessian(cons_i, soadtype, x, Constant(i))
                          for i in 1:num_cons]

        function cons_h!(θ)
            H = map(1:num_cons) do i
                hessian(cons_i, prep_cons_hess[i], soadtype, θ, Constant(i))
            end
            return H
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
            Constant(ones(eltype(x), num_cons)), Constant(p))
        lag_hess_prototype = zeros(Bool, num_cons, length(x))

        function lag_h!(θ, σ, λ)
            if σ == zero(eltype(θ))
                return λ .* cons_h(θ)
            else
                return hessian(lagrangian, lag_prep, soadtype, θ,
                    Constant(σ), Constant(λ), Constant(p))
            end
        end

        if p !== SciMLBase.NullParameters() && p !== nothing
            function lag_h!(θ, σ, λ, p)
                if σ == zero(eltype(θ))
                    return λ .* cons_h(θ)
                else
                    return hessian(lagrangian, lag_prep, soadtype, θ,
                        Constant(σ), Constant(λ), Constant(p))
                end
            end
        end
    elseif lag_h == true && f.cons !== nothing
        lag_h! = (θ, σ, λ, p = p) -> f.lag_h(θ, σ, λ, p)
    else
        lag_h! = nothing
    end

    return OptimizationFunction{false}(f.f, adtype;
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
        cons_expr = f.cons_expr)
end

function instantiate_function(
        f::OptimizationFunction{false}, cache::OptimizationBase.ReInitCache,
        adtype::ADTypes.AbstractADType, num_cons = 0; kwargs...)
    x = cache.u0
    p = cache.p

    return instantiate_function(f, x, adtype, p, num_cons; kwargs...)
end
