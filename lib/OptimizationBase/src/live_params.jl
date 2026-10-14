# `instantiate_live` stores `LiveField` wrappers around the instantiated derivative and
# constraint closures. Each call compares the live `reinit_cache.p` with the `p` the
# closures were built from (one `===`) and re-instantiates once when `reinit!` replaced it.

# The instantiated function, plus what is needed to rebuild it from the live `p`.
mutable struct LiveInstantiation{F, B, AD, RC, P, K}
    f::F           # instantiated function built from `p_src`
    base::B        # un-instantiated function
    adtype::AD
    num_cons::Int
    kwargs::K      # derivative flags forwarded to `instantiate_function`
    rc::RC         # the live `ReInitCache` that `reinit!` mutates
    p_src::P       # the `rc.p` that `f` was built from (compared by `===`)
end

# A data iterator is never handed to `instantiate_function`: the closures are prepared
# with a prototype batch, and minibatch-aware solvers pass the current batch explicitly.
# Otherwise the live `ReInitCache` itself is passed, since some instantiations (the NoAD
# `MultiObjectiveOptimizationFunction` one) read `cache.p` lazily from it.
function _instantiate_from(base, rc, adtype, num_cons, kwargs)
    prep_cache = isa_dataiterator(rc.p) ? ReInitCache(rc.u0, first(iterate(rc.p))) : rc
    return instantiate_function(base, prep_cache, adtype, num_cons; kwargs...)
end

# Solvers declare sparsity structure (and value order, see `lag_hess_structure`) from the
# prototypes seen at `init`, and the cache keeps those. A rebuilt function must therefore
# write the same structure.
_same_structure(a, b) = a === b || _same_structure_(a, b)
function _same_structure_(a::SparseMatrixCSC, b::SparseMatrixCSC)
    ra, ca, _ = findnz(a)
    rb, cb, _ = findnz(b)
    return size(a) == size(b) && ra == rb && ca == cb
end
function _same_structure_(a::AbstractVector{<:AbstractArray}, b::AbstractVector{<:AbstractArray})
    return length(a) == length(b) && all(_same_structure(x, y) for (x, y) in zip(a, b))
end
_same_structure_(a::AbstractArray, b::AbstractArray) = typeof(a) == typeof(b) && size(a) == size(b)
_same_structure_(a, b) = isequal(a, b)

const _STRUCTURE_FIELDS = (
    :hess_prototype, :cons_jac_prototype, :cons_hess_prototype, :lag_hess_prototype,
    :hess_colorvec, :cons_jac_colorvec, :cons_hess_colorvec, :lag_hess_colorvec,
)

@noinline function _structure_error(name)
    throw(
        ArgumentError(
            "`reinit!` changed the parameters in a way that changes the sparsity structure " *
                "of `$name`, which the solver declared at `init`. Build a new cache with " *
                "`init(remake(prob; p = ...), alg)` instead."
        )
    )
end

@noinline function _reinstantiate!(l::LiveInstantiation{F}) where {F}
    p = l.rc.p
    newf = _instantiate_from(l.base, l.rc, l.adtype, l.num_cons, l.kwargs)
    if !(newf isa F)
        throw(
            ArgumentError(
                "`reinit!` changed the parameters in a way that changes the type of the " *
                    "instantiated derivative functions ($(typeof(newf)) vs. $F). Build a new " *
                    "cache with `init(remake(prob; p = ...), alg)` instead."
            )
        )
    end
    for name in _STRUCTURE_FIELDS
        _same_structure(getfield(l.f, name), getfield(newf, name)) || _structure_error(name)
    end
    l.f = newf
    l.p_src = p
    return newf
end

@inline function _current!(l::LiveInstantiation)
    return l.rc.p === l.p_src ? l.f : _reinstantiate!(l)
end

# Callable stored in the `Name` field of a cache's `OptimizationFunction`. `N` is the
# number of arguments of the call form without `p`; `Obj` marks derivatives of the
# objective, which must never be evaluated on a whole data iterator.
struct LiveField{Name, N, Obj, L <: LiveInstantiation}
    live::L
end

@noinline function _iterator_error(name, p)
    throw(
        ArgumentError(
            "The solver evaluated `$name` without selecting a minibatch, but the problem's " *
                "`p` is a data iterator ($(typeof(p))). Only minibatch-aware solvers (e.g. " *
                "OptimizationOptimisers, OptimizationSophia) accept a data iterator as `p`; " *
                "pass the full dataset as `p` for other solvers."
        )
    )
end

# `M` and `N` are compile-time constants, so the branches below fold away.
@inline function (w::LiveField{Name, N, Obj})(args::Vararg{Any, M}) where {Name, N, Obj, M}
    l = w.live
    if M == N
        p = l.rc.p
        Obj && isa_dataiterator(p) && _iterator_error(Name, p)
        return getfield(_current!(l), Name)(args...)
    end
    fn = getfield(_current!(l), Name)
    M == N + 1 || return fn(args...)
    p = last(args)
    Obj && isa_dataiterator(p) && _iterator_error(Name, p)
    # The live `p` is already held by the refreshed closure; dropping it also covers backends
    # whose closures take no `p` (MTK, Zygote constraints). Any other `p` (a minibatch, or the
    # iterator Auglag passes to `cons`) is forwarded.
    if p === l.rc.p && !isa_dataiterator(p)
        return fn(Base.front(args)...)
    end
    return fn(args...)
end

const _LIVE_FIELDS = (
    # name, args without `p` (iip, oop), objective derivative
    (:grad, 2, 1, true),
    (:fg, 2, 1, true),
    (:hess, 2, 1, true),
    (:fgh, 3, 1, true),
    (:hv, 3, 2, true),
    (:lag_h, 4, 3, true),
    (:cons, 2, 1, false),
    (:cons_j, 2, 1, false),
    (:cons_jvp, 3, 2, false),
    (:cons_vjp, 3, 2, false),
    (:cons_h, 2, 1, false),
)

function _wrap_field(fi, live, iip, (name, n_iip, n_oop, obj))
    getfield(fi, name) === nothing && return nothing
    return LiveField{name, iip ? n_iip : n_oop, obj, typeof(live)}(live)
end

function _wrap_live(fi::OptimizationFunction{iip}, live) where {iip}
    w = map(spec -> _wrap_field(fi, live, iip, spec), _LIVE_FIELDS)
    grad, fg, hess, fgh, hv, lag_h, cons, cons_j, cons_jvp, cons_vjp, cons_h = w
    return OptimizationFunction{iip}(
        fi.f, fi.adtype;
        grad, fg, hess, fgh, hv, lag_h, cons, cons_j, cons_jvp, cons_vjp, cons_h,
        hess_prototype = fi.hess_prototype,
        cons_jac_prototype = fi.cons_jac_prototype,
        cons_hess_prototype = fi.cons_hess_prototype,
        observed = fi.observed,
        expr = fi.expr, cons_expr = fi.cons_expr, sys = fi.sys,
        lag_hess_prototype = fi.lag_hess_prototype,
        hess_colorvec = fi.hess_colorvec,
        cons_jac_colorvec = fi.cons_jac_colorvec,
        cons_hess_colorvec = fi.cons_hess_colorvec,
        lag_hess_colorvec = fi.lag_hess_colorvec,
        initialization_data = fi.initialization_data
    )
end

"""
    instantiate_live(f, reinit_cache::ReInitCache, adtype, num_cons = 0; kwargs...)

Like `instantiate_function(f, reinit_cache, adtype, num_cons; kwargs...)`, but the derivative
and constraint closures of the result always use the current `reinit_cache.p`: after
`reinit!(cache; p = ...)` replaces it, the closures are rebuilt on their next call. When
`reinit_cache.p` is a data iterator, the closures are prepared with its first batch, and
calling an objective derivative without an explicit batch throws instead of silently using
that batch.

The prototypes and color vectors of the result are the ones computed at construction, since
solvers declare their sparsity structure from them. If new parameters change that structure,
the next call throws an `ArgumentError`; build a new cache with
`init(remake(prob; p = ...), alg)` in that case.

Solvers that build their own cache should construct their instantiated function with this,
passing the same `ReInitCache` object that the cache's `reinit!` mutates. Returns
`(f_live, f_inst)`, where `f_inst` is the plain instantiated function (for structural
analysis and other one-off uses at construction time).
"""
function instantiate_live(f, rc::ReInitCache, adtype, num_cons = 0; kwargs...)
    kw = NamedTuple(kwargs)
    fi = _instantiate_from(f, rc, adtype, num_cons, kw)
    fi isa OptimizationFunction || return fi, fi  # e.g. MultiObjectiveOptimizationFunction
    live = LiveInstantiation(fi, f, adtype, Int(num_cons), kw, rc, rc.p)
    return _wrap_live(fi, live), fi
end
