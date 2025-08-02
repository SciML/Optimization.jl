
function symbolify(e::Expr)
    if !(e.args[1] isa Symbol)
        e.args[1] = Symbol(e.args[1])
    end
    symbolify.(e.args)
    return e
end

function symbolify(e)
    return e
end

function rep_pars_vals!(e::Expr, p)
    rep_pars_vals!.(e.args, Ref(p))
    replace!(e.args, p...)
end

function rep_pars_vals!(e, p) end

"""
    instantiate_function(f, x, ::AbstractADType, p, num_cons = 0)::OptimizationFunction

This function is used internally by Optimization.jl to construct
the necessary extra functions (gradients, Hessians, etc.) before
optimization. Each of the ADType dispatches use the supplied automatic
differentiation type in order to specify how the construction process
occurs.

If no ADType is given, then the default `NoAD` dispatch simply
defines closures on any supplied gradient function to enclose the
parameters to match the interfaces for the specific optimization
libraries (i.e. (G,x)->f.grad(G,x,p)). If a function is not given
and the `NoAD` dispatch is used, or if the AD dispatch is currently
not capable of defining said derivative, then the constructed
`OptimizationFunction` will simply use `nothing` to specify and undefined
function.

The return of `instantiate_function` is an `OptimizationFunction` which
is then used in the optimization process. If an optimizer requires a
function that is not defined, an error is thrown.

For more information on the use of automatic differentiation, see the
documentation of the `AbstractADType` types.
"""
function OptimizationBase.instantiate_function(
        f::MultiObjectiveOptimizationFunction, x, ::SciMLBase.NoAD,
        p, num_cons = 0; kwargs...)
    jac = f.jac === nothing ? nothing : (J, x, args...) -> f.jac(J, x, p, args...)
    hess = f.hess === nothing ? nothing :
           [(H, x, args...) -> h(H, x, p, args...) for h in f.hess]
    hv = f.hv === nothing ? nothing : (H, x, v, args...) -> f.hv(H, x, v, p, args...)
    cons = f.cons === nothing ? nothing : (res, x) -> f.cons(res, x, p)
    cons_j = f.cons_j === nothing ? nothing : (res, x) -> f.cons_j(res, x, p)
    cons_jvp = f.cons_jvp === nothing ? nothing : (res, x) -> f.cons_jvp(res, x, p)
    cons_vjp = f.cons_vjp === nothing ? nothing : (res, x) -> f.cons_vjp(res, x, p)
    cons_h = f.cons_h === nothing ? nothing : (res, x) -> f.cons_h(res, x, p)
    hess_prototype = f.hess_prototype === nothing ? nothing :
                     convert.(eltype(x), f.hess_prototype)
    cons_jac_prototype = f.cons_jac_prototype === nothing ? nothing :
                         convert.(eltype(x), f.cons_jac_prototype)
    cons_hess_prototype = f.cons_hess_prototype === nothing ? nothing :
                          [convert.(eltype(x), f.cons_hess_prototype[i])
                           for i in 1:num_cons]
    expr = symbolify(f.expr)
    cons_expr = symbolify.(f.cons_expr)

    return MultiObjectiveOptimizationFunction{true}(
        f.f, SciMLBase.NoAD(); jac = jac, hess = hess,
        hv = hv,
        cons = cons, cons_j = cons_j, cons_jvp = cons_jvp, cons_vjp = cons_vjp, cons_h = cons_h,
        hess_prototype = hess_prototype,
        cons_jac_prototype = cons_jac_prototype,
        cons_hess_prototype = cons_hess_prototype,
        expr = expr, cons_expr = cons_expr,
        sys = f.sys,
        observed = f.observed)
end

function OptimizationBase.instantiate_function(
        f::MultiObjectiveOptimizationFunction, cache::ReInitCache, ::SciMLBase.NoAD,
        num_cons = 0; kwargs...)
    jac = f.jac === nothing ? nothing : (J, x, args...) -> f.jac(J, x, cache.p, args...)
    hess = f.hess === nothing ? nothing :
           [(H, x, args...) -> h(H, x, cache.p, args...) for h in f.hess]
    hv = f.hv === nothing ? nothing : (H, x, v, args...) -> f.hv(H, x, v, cache.p, args...)
    cons = f.cons === nothing ? nothing : (res, x) -> f.cons(res, x, cache.p)
    cons_j = f.cons_j === nothing ? nothing : (res, x) -> f.cons_j(res, x, cache.p)
    cons_jvp = f.cons_jvp === nothing ? nothing : (res, x) -> f.cons_jvp(res, x, cache.p)
    cons_vjp = f.cons_vjp === nothing ? nothing : (res, x) -> f.cons_vjp(res, x, cache.p)
    cons_h = f.cons_h === nothing ? nothing : (res, x) -> f.cons_h(res, x, cache.p)
    hess_prototype = f.hess_prototype === nothing ? nothing :
                     convert.(eltype(cache.u0), f.hess_prototype)
    cons_jac_prototype = f.cons_jac_prototype === nothing ? nothing :
                         convert.(eltype(cache.u0), f.cons_jac_prototype)
    cons_hess_prototype = f.cons_hess_prototype === nothing ? nothing :
                          [convert.(eltype(cache.u0), f.cons_hess_prototype[i])
                           for i in 1:num_cons]
    expr = symbolify(f.expr)
    cons_expr = symbolify.(f.cons_expr)

    return MultiObjectiveOptimizationFunction{true}(
        f.f, SciMLBase.NoAD(); jac = jac, hess = hess,
        hv = hv,
        cons = cons, cons_j = cons_j, cons_jvp = cons_jvp, cons_vjp = cons_vjp, cons_h = cons_h,
        hess_prototype = hess_prototype,
        cons_jac_prototype = cons_jac_prototype,
        cons_hess_prototype = cons_hess_prototype,
        expr = expr, cons_expr = cons_expr,
        sys = f.sys,
        observed = f.observed)
end

function OptimizationBase.instantiate_function(
        f::OptimizationFunction{true}, x, ::SciMLBase.NoAD,
        p, num_cons = 0; sense, kwargs...)
    if f.grad === nothing
        grad = nothing
    else
        function grad(G, x)
            return f.grad(G, x, p)
        end
        if p != SciMLBase.NullParameters()
            function grad(G, x, p)
                return f.grad(G, x, p)
            end
        end
    end
    if f.fg === nothing
        fg = nothing
    else
        function fg(G, x)
            return f.fg(G, x, p)
        end
        if p != SciMLBase.NullParameters()
            function fg(G, x, p)
                return f.fg(G, x, p)
            end
        end
    end
    if f.hess === nothing
        hess = nothing
    else
        function hess(H, x)
            return f.hess(H, x, p)
        end
        if p != SciMLBase.NullParameters()
            function hess(H, x, p)
                return f.hess(H, x, p)
            end
        end
    end

    if f.fgh === nothing
        fgh = nothing
    else
        function fgh(G, H, x)
            return f.fgh(G, H, x, p)
        end
        if p != SciMLBase.NullParameters()
            function fgh(G, H, x, p)
                return f.fgh(G, H, x, p)
            end
        end
    end

    if f.hv === nothing
        hv = nothing
    else
        function hv(H, x, v)
            return f.hv(H, x, v, p)
        end
        if p != SciMLBase.NullParameters()
            function hv(H, x, v, p)
                return f.hv(H, x, v, p)
            end
        end
    end

    cons = f.cons === nothing ? nothing : (res, x) -> f.cons(res, x, p)
    cons_j = f.cons_j === nothing ? nothing : (res, x) -> f.cons_j(res, x, p)
    cons_vjp = f.cons_vjp === nothing ? nothing : (res, x) -> f.cons_vjp(res, x, p)
    cons_jvp = f.cons_jvp === nothing ? nothing : (res, x) -> f.cons_jvp(res, x, p)
    cons_h = f.cons_h === nothing ? nothing : (res, x) -> f.cons_h(res, x, p)

    if f.lag_h === nothing
        lag_h = nothing
    else
        function lag_h(res, x)
            return f.lag_h(res, x, p)
        end
        if p != SciMLBase.NullParameters()
            function lag_h(res, x, p)
                return f.lag_h(res, x, p)
            end
        end
    end
    hess_prototype = f.hess_prototype === nothing ? nothing :
                     convert.(eltype(x), f.hess_prototype)
    cons_jac_prototype = f.cons_jac_prototype === nothing ? nothing :
                         convert.(eltype(x), f.cons_jac_prototype)
    cons_hess_prototype = f.cons_hess_prototype === nothing ? nothing :
                          [convert.(eltype(x), f.cons_hess_prototype[i])
                           for i in 1:num_cons]
    expr = symbolify(f.expr)
    cons_expr = symbolify.(f.cons_expr)

    obj_f = (x,p) -> sense == MaxSense ? -1.0* f.f(x,p) : f.f(x,p)
    return OptimizationFunction{true}(obj_f, SciMLBase.NoAD();
        grad = grad, fg = fg, hess = hess, fgh = fgh, hv = hv,
        cons = cons, cons_j = cons_j, cons_h = cons_h,
        cons_vjp = cons_vjp, cons_jvp = cons_jvp,
        lag_h = lag_h,
        hess_prototype = hess_prototype,
        cons_jac_prototype = cons_jac_prototype,
        cons_hess_prototype = cons_hess_prototype,
        expr = expr, cons_expr = cons_expr,
        sys = f.sys,
        observed = f.observed)
end

function OptimizationBase.instantiate_function(
        f::OptimizationFunction{true}, cache::ReInitCache, ::SciMLBase.NoAD,
        num_cons = 0; kwargs...)
    x = cache.u0
    p = cache.p

    return instantiate_function(f, x, SciMLBase.NoAD(), p, num_cons; kwargs...)
end

function instantiate_function(f::OptimizationFunction, x, adtype::ADTypes.AbstractADType,
        p, num_cons = 0; kwargs...)
    adtypestr = string(adtype)
    _strtind = findfirst('.', adtypestr)
    strtind = isnothing(_strtind) ? 5 : _strtind + 5
    open_nrmlbrkt_ind = findfirst('(', adtypestr)
    open_squigllybrkt_ind = findfirst('{', adtypestr)
    open_brkt_ind = isnothing(open_squigllybrkt_ind) ? open_nrmlbrkt_ind :
                    min(open_nrmlbrkt_ind, open_squigllybrkt_ind)
    adpkg = adtypestr[strtind:(open_brkt_ind - 1)]
    throw(ArgumentError("The passed automatic differentiation backend choice is not available. Please load the corresponding AD package $adpkg."))
end
