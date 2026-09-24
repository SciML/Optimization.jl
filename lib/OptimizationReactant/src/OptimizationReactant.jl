"""
    OptimizationReactant

Reactant.jl backend for `OptimizationFunction` instantiation: selecting
`adtype = AutoReactant()` compiles the objective and its derivatives with
[Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) (Enzyme on the traced
program) instead of differentiating the Julia code at run time.

Compiled `grad`/`fg`, Hessian (`h`), Hessian-vector product (`hv`), combined
value/gradient/Hessian (`fgh`), and constraint derivatives (`cons_j`,
`cons_vjp`, `cons_jvp`, `cons_h`, `lag_h`) closures are generated for any
traceable objective — second derivatives are dense and assembled inside the
same compiled program from width-1 forward-over-`<mode>` Hessian-vector
products. `AutoSparse{AutoReactant}` and `SecondOrder{<:AutoReactant}` are
rejected: `AutoReactant` already generates both derivative orders densely,
and sparse detection cannot run through the compiled program.
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

# Second derivatives are built from width-1 forward-over-`<mode>` calls:
# seeding `Duplicated(u, e)` keeps the inner adjoint's padding scalar, whereas
# a batched outer `fwddiff`/`jacobian` widens it and produces invalid MLIR.
function _hvp_in_trace(gcl, u, v, p)
    return only(
        Enzyme.autodiff(
            Enzyme.Forward, gcl, Enzyme.Duplicated, Enzyme.Duplicated(u, v), Const(p)
        )
    )
end

# Dense Hessian as `length(u)` Hessian-vector products over basis vectors.
function _hessian_cols(gcl, u, p)
    es = Enzyme.onehot(u)
    return hcat(ntuple(i -> _hvp_in_trace(gcl, u, es[i], p), length(u))...)
end

function _hessian_objective(f, mode, annot)
    af = annot(f)
    gcl = Const((u, p) -> Enzyme.gradient(mode, af, u, Const(p))[1])
    return (u, p) -> _hessian_cols(gcl, u, p)
end

# Hessian-vector product as the forward-mode derivative of the gradient
# closure seeded with `v`.
function _hvp_objective(f, mode, annot)
    af = annot(f)
    gcl = Const((u, p) -> Enzyme.gradient(mode, af, u, Const(p))[1])
    return (u, v, p) -> _hvp_in_trace(gcl, u, v, p)
end

function _value_grad_hess_objective(f, mode, annot)
    af = annot(f)
    g = (u, p) -> Enzyme.gradient(mode, af, u, Const(p))[1]
    gcl = Const(g)
    return (u, p) -> (f(u, p), g(u, p), _hessian_cols(gcl, u, p))
end

# Out-of-place view of the in-place `cons(res, θ, p)` contract.
function _cons_oop(f, num_cons)
    return (θ, p) -> begin
        res = similar(θ, num_cons)
        f.cons(res, θ, p)
        return res
    end
end

# Constraint Jacobian as `length(θ)` width-1 Jacobian-vector products over
# basis vectors — batched `Enzyme.jacobian` emits `enzyme.extract`/`concat`
# ops that do not export to XLA.
function _cons_j_objective(co)
    cco = Const((θ, p) -> co(θ, p))
    return (θ, p) -> begin
        es = Enzyme.onehot(θ)
        return hcat(
            ntuple(
                i -> only(
                    Enzyme.autodiff(
                        Enzyme.Forward, cco, Enzyme.Duplicated,
                        Enzyme.Duplicated(θ, es[i]), Const(p)
                    )
                ), length(θ)
            )...
        )
    end
end

function _cons_vjp_objective(co, mode)
    return (θ, v, p) -> Enzyme.gradient(
        mode, Const((θ, p) -> sum(v .* co(θ, p))), θ, Const(p)
    )[1]
end

function _cons_jvp_objective(co)
    cco = Const((θ, p) -> co(θ, p))
    return (θ, v, p) -> only(
        Enzyme.autodiff(
            Enzyme.Forward, cco, Enzyme.Duplicated, Enzyme.Duplicated(θ, v), Const(p)
        )
    )
end

function _cons_h_objective(co, mode, num_cons)
    gcls = ntuple(num_cons) do i
        Const(
            (θ, p) -> Enzyme.gradient(
                mode, Const((θ2, p2) -> co(θ2, p2)[i]), θ, Const(p)
            )[1]
        )
    end
    return (θ, p) -> ntuple(i -> _hessian_cols(gcls[i], θ, p), num_cons)
end

function _lag_h_objective(f, co, mode, annot)
    af = annot(f)
    return (θ, σ, λ, p) -> begin
        # `sum(λ .* c)` fuses into a `stablehlo.reduce` whose adjoint is
        # unsupported under nested autodiff; the unrolled product-sum avoids it.
        L = Const(
            (θ2, p2) -> begin
                c = co(θ2, p2)
                s = σ * f(θ2, p2)
                for i in eachindex(c)
                    s = s + λ[i] * c[i]
                end
                return s
            end
        )
        gL = Const((θ2, p2) -> Enzyme.gradient(mode, L, θ2, Const(p2))[1])
        return _hessian_cols(gL, θ, p)
    end
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

    co = f.cons === nothing ? nothing : _cons_oop(f, num_cons)
    cc_h = h == true && f.hess === nothing ?
        CompiledCall(_hessian_objective(f.f, mode, annot)) : nothing
    cc_hv = hv == true && f.hv === nothing ?
        CompiledCall(_hvp_objective(f.f, mode, annot)) : nothing
    cc_fgh = fgh == true && f.fgh === nothing ?
        CompiledCall(_value_grad_hess_objective(f.f, mode, annot)) : nothing
    cc_cj = cons_j == true && co !== nothing && f.cons_j === nothing ?
        CompiledCall(_cons_j_objective(co)) : nothing
    cc_cvjp = cons_vjp == true && co !== nothing && f.cons_vjp === nothing ?
        CompiledCall(_cons_vjp_objective(co, mode)) : nothing
    cc_cjvp = cons_jvp == true && co !== nothing && f.cons_jvp === nothing ?
        CompiledCall(_cons_jvp_objective(co)) : nothing
    cc_ch = cons_h == true && co !== nothing && f.cons_h === nothing ?
        CompiledCall(_cons_h_objective(co, mode, num_cons)) : nothing
    cc_lag = lag_h == true && co !== nothing && f.lag_h === nothing ?
        CompiledCall(_lag_h_objective(f.f, co, mode, annot)) : nothing

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
        let cc_h = cc_h, p = p
            (res, θ, p = p) -> _copyinto!(res, cc_h(θ, p))
        end
    elseif h == true
        let f = f, p = p
            (res, θ, p = p) -> f.hess(res, θ, p)
        end
    else
        nothing
    end

    hv! = if hv == true && f.hv === nothing
        let cc_hv = cc_hv, p = p
            (res, θ, v, p = p) -> _copyinto!(res, cc_hv(θ, v, p))
        end
    elseif hv == true
        let f = f, p = p
            (res, θ, v, p = p) -> f.hv(res, θ, v, p)
        end
    else
        nothing
    end

    fgh! = if fgh == true && f.fgh === nothing
        let cc_fgh = cc_fgh, p = p
            function (G, H, θ, p = p)
                y, gres, hres = cc_fgh(θ, p)
                _copyinto!(G, gres)
                _copyinto!(H, hres)
                return _scalar(y)
            end
        end
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
        let cc_cj = cc_cj, p = p
            (J, θ, p = p) -> _copyinto!(J, cc_cj(θ, p))
        end
    elseif cons_j == true && f.cons !== nothing
        let f = f, p = p
            (J, θ, p = p) -> f.cons_j(J, θ, p)
        end
    else
        nothing
    end

    cons_vjp! = if cons_vjp == true && f.cons !== nothing && f.cons_vjp === nothing
        let cc_cvjp = cc_cvjp, p = p
            (J, θ, v) -> _copyinto!(J, cc_cvjp(θ, v, p))
        end
    elseif cons_vjp == true && f.cons !== nothing
        let f = f, p = p
            (J, θ, v) -> f.cons_vjp(J, θ, v, p)
        end
    else
        nothing
    end

    cons_jvp! = if cons_jvp == true && f.cons !== nothing && f.cons_jvp === nothing
        let cc_cjvp = cc_cjvp, p = p
            (J, θ, v) -> _copyinto!(J, cc_cjvp(θ, v, p))
        end
    elseif cons_jvp == true && f.cons !== nothing
        let f = f, p = p
            (J, θ, v) -> f.cons_jvp(J, θ, v, p)
        end
    else
        nothing
    end

    cons_h! = if cons_h == true && f.cons !== nothing && f.cons_h === nothing
        let cc_ch = cc_ch, p = p, num_cons = num_cons
            function (res, θ)
                Hs = cc_ch(θ, p)
                for i in 1:num_cons
                    _copyinto!(res[i], Hs[i])
                end
                return res
            end
        end
    elseif cons_h == true && f.cons !== nothing
        let f = f, p = p
            (res, θ) -> f.cons_h(res, θ, p)
        end
    else
        nothing
    end

    lag_h! = if lag_h == true && f.cons !== nothing && f.lag_h === nothing
        let cc_lag = cc_lag, p = p
            function (res, θ, σ, λ, p = p)
                M = cc_lag(θ, σ, λ, p)
                if res isa AbstractVector
                    H = Array(M)
                    k = 0
                    for i in 1:length(θ), j in 1:i
                        k += 1
                        res[k] = H[i, j]
                    end
                    return res
                end
                return _copyinto!(res, M)
            end
        end
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
    cc_h !== nothing && cc_h(x, p)
    cc_hv !== nothing && cc_hv(x, x, p)
    cc_fgh !== nothing && cc_fgh(x, p)
    cc_cj !== nothing && cc_cj(x, p)
    cc_cvjp !== nothing && cc_cvjp(x, zeros(eltype(x), num_cons), p)
    cc_cjvp !== nothing && cc_cjvp(x, x, p)
    cc_ch !== nothing && cc_ch(x, p)
    cc_lag !== nothing &&
        cc_lag(x, one(eltype(x)), zeros(eltype(x), num_cons), p)

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
        lag_hess_prototype = lag_h! === nothing ? f.lag_hess_prototype :
            something(f.lag_hess_prototype, zeros(Bool, length(x), length(x))),
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

    cc_h = h == true && f.hess === nothing ?
        CompiledCall(_hessian_objective(f.f, mode, annot)) : nothing
    cc_hv = hv == true && f.hv === nothing ?
        CompiledCall(_hvp_objective(f.f, mode, annot)) : nothing
    cc_fgh = fgh == true && f.fgh === nothing ?
        CompiledCall(_value_grad_hess_objective(f.f, mode, annot)) : nothing
    cc_cj = cons_j == true && f.cons !== nothing && f.cons_j === nothing ?
        CompiledCall(_cons_j_objective(f.cons)) : nothing
    cc_cvjp = cons_vjp == true && f.cons !== nothing && f.cons_vjp === nothing ?
        CompiledCall(_cons_vjp_objective(f.cons, mode)) : nothing
    cc_cjvp = cons_jvp == true && f.cons !== nothing && f.cons_jvp === nothing ?
        CompiledCall(_cons_jvp_objective(f.cons)) : nothing
    cc_ch = cons_h == true && f.cons !== nothing && f.cons_h === nothing ?
        CompiledCall(_cons_h_objective(f.cons, mode, num_cons)) : nothing
    cc_lag = lag_h == true && f.cons !== nothing && f.lag_h === nothing ?
        CompiledCall(_lag_h_objective(f.f, f.cons, mode, annot)) : nothing

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
        let cc_h = cc_h, p = p
            (θ, p = p) -> _out(cc_h(θ, p), θ)
        end
    elseif h == true
        let f = f, p = p
            (θ, p = p) -> f.hess(θ, p)
        end
    else
        nothing
    end

    hv! = if hv == true && f.hv === nothing
        let cc_hv = cc_hv, p = p
            (θ, v, p = p) -> _out(cc_hv(θ, v, p), θ)
        end
    elseif hv == true
        let f = f, p = p
            (θ, v, p = p) -> f.hv(θ, v, p)
        end
    else
        nothing
    end

    fgh! = if fgh == true && f.fgh === nothing
        let cc_fgh = cc_fgh, p = p
            function (θ, p = p)
                y, gres, hres = cc_fgh(θ, p)
                return _scalar(y), _out(gres, θ), _out(hres, θ)
            end
        end
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
        let cc_cj = cc_cj, p = p
            (θ, p = p) -> _out(cc_cj(θ, p), θ)
        end
    elseif cons_j == true && f.cons !== nothing
        let f = f, p = p
            (θ, p = p) -> f.cons_j(θ, p)
        end
    else
        nothing
    end

    cons_vjp! = if cons_vjp == true && f.cons !== nothing && f.cons_vjp === nothing
        let cc_cvjp = cc_cvjp, p = p
            (θ, v) -> _out(cc_cvjp(θ, v, p), θ)
        end
    elseif cons_vjp == true && f.cons !== nothing
        let f = f, p = p
            (θ, v) -> f.cons_vjp(θ, v, p)
        end
    else
        nothing
    end

    cons_jvp! = if cons_jvp == true && f.cons !== nothing && f.cons_jvp === nothing
        let cc_cjvp = cc_cjvp, p = p
            (θ, v) -> _out(cc_cjvp(θ, v, p), θ)
        end
    elseif cons_jvp == true && f.cons !== nothing
        let f = f, p = p
            (θ, v) -> f.cons_jvp(θ, v, p)
        end
    else
        nothing
    end

    cons_h! = if cons_h == true && f.cons !== nothing && f.cons_h === nothing
        let cc_ch = cc_ch, p = p
            (θ) -> [_out(M, θ) for M in cc_ch(θ, p)]
        end
    elseif cons_h == true && f.cons !== nothing
        let f = f, p = p
            (θ) -> f.cons_h(θ, p)
        end
    else
        nothing
    end

    lag_h! = if lag_h == true && f.cons !== nothing && f.lag_h === nothing
        let cc_lag = cc_lag, p = p
            (θ, σ, λ, p = p) -> _out(cc_lag(θ, σ, λ, p), θ)
        end
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
    cc_h !== nothing && cc_h(x, p)
    cc_hv !== nothing && cc_hv(x, x, p)
    cc_fgh !== nothing && cc_fgh(x, p)
    cc_cj !== nothing && cc_cj(x, p)
    cc_cvjp !== nothing && cc_cvjp(x, zeros(eltype(x), num_cons), p)
    cc_cjvp !== nothing && cc_cjvp(x, x, p)
    cc_ch !== nothing && cc_ch(x, p)
    cc_lag !== nothing &&
        cc_lag(x, one(eltype(x)), zeros(eltype(x), num_cons), p)

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
        lag_hess_prototype = lag_h! === nothing ? f.lag_hess_prototype :
            something(f.lag_hess_prototype, zeros(Bool, length(x), length(x))),
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
