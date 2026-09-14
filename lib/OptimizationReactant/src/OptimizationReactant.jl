"""
    OptimizationReactant

Reactant.jl backend for `OptimizationFunction` instantiation: selecting
`adtype = AutoReactant()` compiles the objective and its derivatives with
[Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) (Enzyme on the traced
program) instead of differentiating the Julia code at run time.

Compiled `grad`/`fg` closures are provided for any traceable objective.
Hessians, Hessian-vector products and constraint derivatives are not
generated yet; pass them explicitly to `OptimizationFunction` or choose a
different `adtype`. `AutoSparse{AutoReactant}` and `SecondOrder{<:AutoReactant}`
are rejected.
"""
module OptimizationReactant

import OptimizationBase: ReInitCache, instantiate_function
using ADTypes: ADTypes, AutoEnzyme, AutoReactant, AutoSparse
using DifferentiationInterface: SecondOrder
using Enzyme: Enzyme, Const
using GPUArraysCore: GPUArraysCore
using Reactant: Reactant
using SciMLBase: SciMLBase, OptimizationFunction

export AutoReactant

_to_rarray(x) = Reactant.to_rarray(x; track_numbers = AbstractFloat)

"""
    CompiledCall(f)

Callable that dispatches `f` to a `Reactant.compile`d program specialized on
the (converted) argument types. Arguments are moved to the device with
`Reactant.to_rarray` on every call — `AbstractFloat` leaves are tracked so
scalar parameters stay runtime inputs instead of being baked as constants —
the first call for a given signature compiles (and runs) the program, and
later calls with the same signature reuse it. Passing host `Array`s keeps
working but pays a host-to-device copy per call; place `u0`/`p` on the device
once with `Reactant.to_rarray` to avoid it.
"""
struct CompiledCall{F}
    f::F
    thunks::Dict{Any, Any}
end

CompiledCall(f::F) where {F} = CompiledCall{F}(f, Dict{Any, Any}())

function (c::CompiledCall)(args...)
    rargs = map(_to_rarray, args)
    thunk = get!(c.thunks, map(typeof, rargs)) do
        # Scalar `x[i]` indexing is routine in user objectives; inside the
        # trace it becomes a gather/slice op, which is correct, just slower.
        return GPUArraysCore.allowscalar(() -> Reactant.compile(c.f, rargs))
    end
    return thunk(rargs...)
end

_scalar(x::Reactant.AbstractConcreteNumber) = Reactant.to_number(x)
_scalar(x::Reactant.AbstractConcreteArray{<:Any, 0}) = only(Array(x))
_scalar(x) = x

# `grad` writes into `res`, so the destination decides the transfer direction.
function _copyinto!(res::Reactant.AbstractConcreteArray, g::Reactant.AbstractConcreteArray)
    copyto!(res, g)
    return res
end
function _copyinto!(res::Reactant.AbstractConcreteArray, g)
    copyto!(res, g)
    return res
end
function _copyinto!(res, g::Reactant.AbstractConcreteArray)
    copyto!(res, Array(g))
    return res
end
function _copyinto!(res, g)
    copyto!(res, g)
    return res
end

# `{false}` gradients return a new array matching the input's placement.
_out(g::Reactant.AbstractConcreteArray, θ::Reactant.AbstractConcreteArray) = g
_out(g::Reactant.AbstractConcreteArray, θ) = Array(g)
_out(g, θ) = g

_mode(adtype::AutoReactant) = _mode(adtype.mode)
_mode(ad::AutoEnzyme) = ad.mode === nothing ? Enzyme.Reverse : ad.mode
_annot(adtype::AutoReactant) = _annot(adtype.mode)
_annot(::AutoEnzyme{<:Any, A}) where {A} = A <: Nothing ? Const : A

# Reverse-mode gradient of `f` w.r.t. `u`, differentiated inside the compiled
# program: the Enzyme call is traced and lowered to the StableHLO level, so no
# Julia-level activity analysis ever sees the parameter buffers.
function _grad_objective(f, mode, annot)
    af = annot(f)
    return (u, p) -> Enzyme.gradient(mode, af, u, Const(p))[1]
end

function _value_grad_objective(f, mode, annot)
    af = annot(f)
    return (u, p) -> (f(u, p), Enzyme.gradient(mode, af, u, Const(p))[1])
end

_unsupported(what) = throw(
    ArgumentError(
        "`AutoReactant` cannot generate $what yet; pass it explicitly to \
        `OptimizationFunction` or choose a different `adtype`."
    )
)

const _REACTANT_UNSUPPORTED_ADTYPE = Union{
    AutoSparse{<:AutoReactant}, AutoSparse{<:SecondOrder{<:AutoReactant}},
    SecondOrder{<:AutoReactant}, SecondOrder{<:Any, <:AutoReactant},
}

for iip in (true, false)
    @eval function instantiate_function(
            ::OptimizationFunction{$iip}, ::Any,
            ::_REACTANT_UNSUPPORTED_ADTYPE, ::Any, ::Any = 0; kwargs...
        )
        return _unsupported("sparse or second-order derivative schemes")
    end
    @eval function instantiate_function(
            ::OptimizationFunction{$iip}, ::ReInitCache,
            ::_REACTANT_UNSUPPORTED_ADTYPE, ::Any = 0; kwargs...
        )
        return _unsupported("sparse or second-order derivative schemes")
    end
end

function instantiate_function(
        f::OptimizationFunction{true}, x, adtype::AutoReactant,
        p = SciMLBase.NullParameters(), num_cons = 0;
        g = false, h = false, hv = false, fg = false, fgh = false,
        cons_j = false, cons_vjp = false, cons_jvp = false, cons_h = false,
        lag_h = false
    )
    mode = _mode(adtype)
    annot = _annot(adtype)

    cc_f = CompiledCall(f.f)
    cc_g = CompiledCall(_grad_objective(f.f, mode, annot))
    cc_fg = CompiledCall(_value_grad_objective(f.f, mode, annot))

    fnew = let cc_f = cc_f, p = p
        (θ, p = p) -> _scalar(cc_f(θ, p))
    end

    grad = if g == true && f.grad === nothing
        let cc_g = cc_g, p = p
            (res, θ, p = p) -> _copyinto!(res, cc_g(θ, p))
        end
    elseif g == true
        let f = f, p = p
            (res, θ, p = p) -> f.grad(res, θ, p)
        end
    else
        nothing
    end

    fg! = if fg == true && f.fg === nothing && f.grad !== nothing
        let f = f, p = p
            function (res, θ, p = p)
                f.grad(res, θ, p)
                return f.f(θ, p)
            end
        end
    elseif fg == true && f.fg === nothing
        let cc_fg = cc_fg, p = p
            function (res, θ, p = p)
                y, gres = cc_fg(θ, p)
                _copyinto!(res, gres)
                return _scalar(y)
            end
        end
    elseif fg == true
        let f = f, p = p
            (res, θ, p = p) -> f.fg(res, θ, p)
        end
    else
        nothing
    end

    hess = if h == true && f.hess === nothing
        _unsupported("Hessians")
    elseif h == true
        let f = f, p = p
            (res, θ, p = p) -> f.hess(res, θ, p)
        end
    else
        nothing
    end

    hv! = if hv == true && f.hv === nothing
        _unsupported("Hessian-vector products")
    elseif hv == true
        let f = f, p = p
            (res, θ, v, p = p) -> f.hv(res, θ, v, p)
        end
    else
        nothing
    end

    fgh! = if fgh == true && f.fgh === nothing
        _unsupported("combined objective/gradient/Hessian evaluation")
    elseif fgh == true
        let f = f, p = p
            (G, H, θ, p = p) -> f.fgh(G, H, θ, p)
        end
    else
        nothing
    end

    cons = if f.cons !== nothing
        let f = f, p = p
            (res, θ, p_call = p) -> f.cons(res, θ, p_call)
        end
    else
        nothing
    end

    cons_j! = if cons_j == true && f.cons !== nothing && f.cons_j === nothing
        _unsupported("constraint Jacobians")
    elseif cons_j == true && f.cons !== nothing
        let f = f, p = p
            (J, θ, p = p) -> f.cons_j(J, θ, p)
        end
    else
        nothing
    end

    cons_vjp! = if cons_vjp == true && f.cons !== nothing && f.cons_vjp === nothing
        _unsupported("constraint vector-Jacobian products")
    elseif cons_vjp == true && f.cons !== nothing
        let f = f, p = p
            (J, θ, v) -> f.cons_vjp(J, θ, v, p)
        end
    else
        nothing
    end

    cons_jvp! = if cons_jvp == true && f.cons !== nothing && f.cons_jvp === nothing
        _unsupported("constraint Jacobian-vector products")
    elseif cons_jvp == true && f.cons !== nothing
        let f = f, p = p
            (J, θ, v) -> f.cons_jvp(J, θ, v, p)
        end
    else
        nothing
    end

    cons_h! = if cons_h == true && f.cons !== nothing && f.cons_h === nothing
        _unsupported("constraint Hessians")
    elseif cons_h == true && f.cons !== nothing
        let f = f, p = p
            (res, θ) -> f.cons_h(res, θ, p)
        end
    else
        nothing
    end

    lag_h! = if lag_h == true && f.cons !== nothing && f.lag_h === nothing
        _unsupported("Lagrangian Hessians")
    elseif lag_h == true && f.cons !== nothing
        let f = f, p = p
            (res, θ, σ, μ, p = p) -> f.lag_h(res, θ, σ, μ, p)
        end
    else
        nothing
    end

    # Compile eagerly against the construction types so an untraceable
    # objective fails at `init` rather than mid-solve.
    cc_f(x, p)
    (g == true && f.grad === nothing) && cc_g(x, p)
    (fg == true && f.fg === nothing) && cc_fg(x, p)

    return OptimizationFunction{true}(
        fnew, adtype;
        grad = grad, fg = fg!, hess = hess, hv = hv!, fgh = fgh!,
        cons = cons, cons_j = cons_j!, cons_h = cons_h!,
        cons_vjp = cons_vjp!, cons_jvp = cons_jvp!,
        hess_prototype = f.hess_prototype,
        hess_colorvec = f.hess_colorvec,
        cons_jac_prototype = f.cons_jac_prototype,
        cons_jac_colorvec = f.cons_jac_colorvec,
        cons_hess_prototype = f.cons_hess_prototype,
        cons_hess_colorvec = f.cons_hess_colorvec,
        lag_h = lag_h!,
        lag_hess_prototype = f.lag_hess_prototype,
        sys = f.sys,
        expr = f.expr,
        cons_expr = f.cons_expr,
        observed = f.observed
    )
end

function instantiate_function(
        f::OptimizationFunction{true}, cache::ReInitCache,
        adtype::AutoReactant, num_cons = 0; kwargs...
    )
    return instantiate_function(f, cache.u0, adtype, cache.p, num_cons; kwargs...)
end

function instantiate_function(
        f::OptimizationFunction{false}, x, adtype::AutoReactant,
        p = SciMLBase.NullParameters(), num_cons = 0;
        g = false, h = false, hv = false, fg = false, fgh = false,
        cons_j = false, cons_vjp = false, cons_jvp = false, cons_h = false,
        lag_h = false
    )
    mode = _mode(adtype)
    annot = _annot(adtype)

    cc_f = CompiledCall(f.f)
    cc_g = CompiledCall(_grad_objective(f.f, mode, annot))
    cc_fg = CompiledCall(_value_grad_objective(f.f, mode, annot))

    fnew = let cc_f = cc_f, p = p
        (θ, p = p) -> _scalar(cc_f(θ, p))
    end

    grad = if g == true && f.grad === nothing
        let cc_g = cc_g, p = p
            (θ, p = p) -> _out(cc_g(θ, p), θ)
        end
    elseif g == true
        let f = f, p = p
            (θ, p = p) -> f.grad(θ, p)
        end
    else
        nothing
    end

    fg! = if fg == true && f.fg === nothing && f.grad !== nothing
        let f = f, p = p
            (θ, p = p) -> (f.f(θ, p), f.grad(θ, p))
        end
    elseif fg == true && f.fg === nothing
        let cc_fg = cc_fg, p = p
            function (θ, p = p)
                y, gres = cc_fg(θ, p)
                return _scalar(y), _out(gres, θ)
            end
        end
    elseif fg == true
        let f = f, p = p
            (θ, p = p) -> f.fg(θ, p)
        end
    else
        nothing
    end

    hess = if h == true && f.hess === nothing
        _unsupported("Hessians")
    elseif h == true
        let f = f, p = p
            (θ, p = p) -> f.hess(θ, p)
        end
    else
        nothing
    end

    hv! = if hv == true && f.hv === nothing
        _unsupported("Hessian-vector products")
    elseif hv == true
        let f = f, p = p
            (θ, v, p = p) -> f.hv(θ, v, p)
        end
    else
        nothing
    end

    fgh! = if fgh == true && f.fgh === nothing
        _unsupported("combined objective/gradient/Hessian evaluation")
    elseif fgh == true
        let f = f, p = p
            (θ, p = p) -> f.fgh(θ, p)
        end
    else
        nothing
    end

    cons = if f.cons !== nothing
        let f = f, p = p
            (θ, p_call = p) -> f.cons(θ, p_call)
        end
    else
        nothing
    end

    cons_j! = if cons_j == true && f.cons !== nothing && f.cons_j === nothing
        _unsupported("constraint Jacobians")
    elseif cons_j == true && f.cons !== nothing
        let f = f, p = p
            (θ, p = p) -> f.cons_j(θ, p)
        end
    else
        nothing
    end

    cons_vjp! = if cons_vjp == true && f.cons !== nothing && f.cons_vjp === nothing
        _unsupported("constraint vector-Jacobian products")
    elseif cons_vjp == true && f.cons !== nothing
        let f = f, p = p
            (θ, v) -> f.cons_vjp(θ, v, p)
        end
    else
        nothing
    end

    cons_jvp! = if cons_jvp == true && f.cons !== nothing && f.cons_jvp === nothing
        _unsupported("constraint Jacobian-vector products")
    elseif cons_jvp == true && f.cons !== nothing
        let f = f, p = p
            (θ, v) -> f.cons_jvp(θ, v, p)
        end
    else
        nothing
    end

    cons_h! = if cons_h == true && f.cons !== nothing && f.cons_h === nothing
        _unsupported("constraint Hessians")
    elseif cons_h == true && f.cons !== nothing
        let f = f, p = p
            (θ) -> f.cons_h(θ, p)
        end
    else
        nothing
    end

    lag_h! = if lag_h == true && f.cons !== nothing && f.lag_h === nothing
        _unsupported("Lagrangian Hessians")
    elseif lag_h == true && f.cons !== nothing
        let f = f, p = p
            (θ, σ, λ, p = p) -> f.lag_h(θ, σ, λ, p)
        end
    else
        nothing
    end

    cc_f(x, p)
    (g == true && f.grad === nothing) && cc_g(x, p)
    (fg == true && f.fg === nothing) && cc_fg(x, p)

    return OptimizationFunction{false}(
        fnew, adtype;
        grad = grad, fg = fg!, hess = hess, hv = hv!, fgh = fgh!,
        cons = cons, cons_j = cons_j!, cons_h = cons_h!,
        cons_vjp = cons_vjp!, cons_jvp = cons_jvp!,
        hess_prototype = f.hess_prototype,
        hess_colorvec = f.hess_colorvec,
        cons_jac_prototype = f.cons_jac_prototype,
        cons_jac_colorvec = f.cons_jac_colorvec,
        cons_hess_prototype = f.cons_hess_prototype,
        cons_hess_colorvec = f.cons_hess_colorvec,
        lag_h = lag_h!,
        lag_hess_prototype = f.lag_hess_prototype,
        sys = f.sys,
        expr = f.expr,
        cons_expr = f.cons_expr,
        observed = f.observed
    )
end

function instantiate_function(
        f::OptimizationFunction{false}, cache::ReInitCache,
        adtype::AutoReactant, num_cons = 0; kwargs...
    )
    return instantiate_function(f, cache.u0, adtype, cache.p, num_cons; kwargs...)
end

end
