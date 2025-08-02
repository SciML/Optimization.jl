module OptimizationMTKExt

import OptimizationBase, OptimizationBase.ArrayInterface
import OptimizationBase.SciMLBase
import OptimizationBase.SciMLBase: OptimizationFunction
import OptimizationBase.ADTypes: AutoModelingToolkit, AutoSymbolics, AutoSparse
using ModelingToolkit

function OptimizationBase.instantiate_function(
        f::OptimizationFunction{true}, x, adtype::AutoSparse{<:AutoSymbolics}, p,
        num_cons = 0;
        g = false, h = false, hv = false, fg = false, fgh = false,
        cons_j = false, cons_vjp = false, cons_jvp = false, cons_h = false,
        lag_h = false)
    p = isnothing(p) ? SciMLBase.NullParameters() : p

    sys = complete(ModelingToolkit.modelingtoolkitize(OptimizationProblem(f, x, p;
        lcons = fill(0.0,
            num_cons),
        ucons = fill(0.0,
            num_cons))))
    #sys = ModelingToolkit.structural_simplify(sys)
    # don't need to pass `x` or `p` since they're defaults now
    f = OptimizationProblem(sys, nothing; grad = g, hess = h,
        sparse = true, cons_j = cons_j, cons_h = cons_h,
        cons_sparse = true).f

    grad = (G, θ, args...) -> f.grad(G, θ, p, args...)

    hess = (H, θ, args...) -> f.hess(H, θ, p, args...)

    hv = function (H, θ, v, args...)
        res = (eltype(θ)).(f.hess_prototype)
        hess(res, θ, args...)
        H .= res * v
    end

    if !isnothing(f.cons)
        cons = (res, θ) -> f.cons(res, θ, p)
        cons_j = (J, θ) -> f.cons_j(J, θ, p)
        cons_h = (res, θ) -> f.cons_h(res, θ, p)
    else
        cons = nothing
        cons_j = nothing
        cons_h = nothing
    end

    return OptimizationFunction{true}(f.f, adtype; grad = grad, hess = hess, hv = hv,
        cons = cons, cons_j = cons_j, cons_h = cons_h,
        hess_prototype = f.hess_prototype,
        cons_jac_prototype = f.cons_jac_prototype,
        cons_hess_prototype = f.cons_hess_prototype,
        expr = OptimizationBase.symbolify(f.expr),
        cons_expr = OptimizationBase.symbolify.(f.cons_expr),
        sys = sys,
        observed = f.observed)
end

function OptimizationBase.instantiate_function(
        f::OptimizationFunction{true}, cache::OptimizationBase.ReInitCache,
        adtype::AutoSparse{<:AutoSymbolics}, num_cons = 0;
        g = false, h = false, hv = false, fg = false, fgh = false,
        cons_j = false, cons_vjp = false, cons_jvp = false, cons_h = false,
        lag_h = false)
    p = isnothing(cache.p) ? SciMLBase.NullParameters() : cache.p

    sys = complete(ModelingToolkit.modelingtoolkitize(OptimizationProblem(f, cache.u0,
        cache.p;
        lcons = fill(0.0,
            num_cons),
        ucons = fill(0.0,
            num_cons))))
    #sys = ModelingToolkit.structural_simplify(sys)
    # don't need to pass `x` or `p` since they're defaults now
    f = OptimizationProblem(sys, nothing; grad = g, hess = h,
        sparse = true, cons_j = cons_j, cons_h = cons_h,
        cons_sparse = true).f

    grad = (G, θ, args...) -> f.grad(G, θ, cache.p, args...)

    hess = (H, θ, args...) -> f.hess(H, θ, cache.p, args...)

    hv = function (H, θ, v, args...)
        res = (eltype(θ)).(f.hess_prototype)
        hess(res, θ, args...)
        H .= res * v
    end

    if !isnothing(f.cons)
        cons = (res, θ) -> f.cons(res, θ, cache.p)
        cons_j = (J, θ) -> f.cons_j(J, θ, cache.p)
        cons_h = (res, θ) -> f.cons_h(res, θ, cache.p)
    else
        cons = nothing
        cons_j = nothing
        cons_h = nothing
    end

    return OptimizationFunction{true}(f.f, adtype; grad = grad, hess = hess, hv = hv,
        cons = cons, cons_j = cons_j, cons_h = cons_h,
        hess_prototype = f.hess_prototype,
        cons_jac_prototype = f.cons_jac_prototype,
        cons_hess_prototype = f.cons_hess_prototype,
        expr = OptimizationBase.symbolify(f.expr),
        cons_expr = OptimizationBase.symbolify.(f.cons_expr),
        sys = sys,
        observed = f.observed)
end

function OptimizationBase.instantiate_function(
        f::OptimizationFunction{true}, x, adtype::AutoSymbolics, p,
        num_cons = 0; g = false, h = false, hv = false, fg = false, fgh = false,
        cons_j = false, cons_vjp = false, cons_jvp = false, cons_h = false,
        lag_h = false)
    p = isnothing(p) ? SciMLBase.NullParameters() : p

    sys = complete(ModelingToolkit.modelingtoolkitize(OptimizationProblem(f, x, p;
        lcons = fill(0.0,
            num_cons),
        ucons = fill(0.0,
            num_cons))))
    #sys = ModelingToolkit.structural_simplify(sys)
    # don't need to pass `x` or `p` since they're defaults now
    f = OptimizationProblem(sys, nothing; grad = g, hess = h,
        sparse = false, cons_j = cons_j, cons_h = cons_h,
        cons_sparse = false).f

    grad = (G, θ, args...) -> f.grad(G, θ, p, args...)

    hess = (H, θ, args...) -> f.hess(H, θ, p, args...)

    hv = function (H, θ, v, args...)
        res = ArrayInterface.zeromatrix(θ)
        hess(res, θ, args...)
        H .= res * v
    end

    if !isnothing(f.cons)
        cons = (res, θ) -> f.cons(res, θ, p)
        cons_j = (J, θ) -> f.cons_j(J, θ, p)
        cons_h = (res, θ) -> f.cons_h(res, θ, p)
    else
        cons = nothing
        cons_j = nothing
        cons_h = nothing
    end

    return OptimizationFunction{true}(f.f, adtype; grad = grad, hess = hess, hv = hv,
        cons = cons, cons_j = cons_j, cons_h = cons_h,
        hess_prototype = f.hess_prototype,
        cons_jac_prototype = f.cons_jac_prototype,
        cons_hess_prototype = f.cons_hess_prototype,
        expr = OptimizationBase.symbolify(f.expr),
        cons_expr = OptimizationBase.symbolify.(f.cons_expr),
        sys = sys,
        observed = f.observed)
end

function OptimizationBase.instantiate_function(
        f::OptimizationFunction{true}, cache::OptimizationBase.ReInitCache,
        adtype::AutoSymbolics, num_cons = 0;
        g = false, h = false, hv = false, fg = false, fgh = false,
        cons_j = false, cons_vjp = false, cons_jvp = false, cons_h = false,
        lag_h = false)
    p = isnothing(cache.p) ? SciMLBase.NullParameters() : cache.p

    sys = complete(ModelingToolkit.modelingtoolkitize(OptimizationProblem(f, cache.u0,
        cache.p;
        lcons = fill(0.0,
            num_cons),
        ucons = fill(0.0,
            num_cons))))
    #sys = ModelingToolkit.structural_simplify(sys)
    # don't need to pass `x` or `p` since they're defaults now
    f = OptimizationProblem(sys, nothing; grad = g, hess = h,
        sparse = false, cons_j = cons_j, cons_h = cons_h,
        cons_sparse = false).f

    grad = (G, θ, args...) -> f.grad(G, θ, cache.p, args...)

    hess = (H, θ, args...) -> f.hess(H, θ, cache.p, args...)

    hv = function (H, θ, v, args...)
        res = ArrayInterface.zeromatrix(θ)
        hess(res, θ, args...)
        H .= res * v
    end

    if !isnothing(f.cons)
        cons = (res, θ) -> f.cons(res, θ, cache.p)
        cons_j = (J, θ) -> f.cons_j(J, θ, cache.p)
        cons_h = (res, θ) -> f.cons_h(res, θ, cache.p)
    else
        cons = nothing
        cons_j = nothing
        cons_h = nothing
    end

    return OptimizationFunction{true}(f.f, adtype; grad = grad, hess = hess, hv = hv,
        cons = cons, cons_j = cons_j, cons_h = cons_h,
        hess_prototype = f.hess_prototype,
        cons_jac_prototype = f.cons_jac_prototype,
        cons_hess_prototype = f.cons_hess_prototype,
        expr = OptimizationBase.symbolify(f.expr),
        cons_expr = OptimizationBase.symbolify.(f.cons_expr),
        sys = sys,
        observed = f.observed)
end

end
