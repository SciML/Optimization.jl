# Type erasure for `OptimizationFunction{iip, AutoSpecialize}` and
# `OptimizationFunction{iip, AutoDespecialize}`.
#
# The solvers build closures over their caches, and the caches hold the objective, its
# derivatives, the constraints and `p`. With full specialization each of those types is
# repeated at every layer and the solver code is compiled anew for every objective. Here
# the functions are stored behind a field of type `Any` before `instantiate_function`
# runs, so the AD preparations, the instantiated closures and the solver cache have the
# same types for every objective. `AutoDespecialize` also stores `p` in a
# `SciMLBase.DespecializedParameters`. The cost is one dynamic dispatch per call, below
# which the user's function stays fully specialized.
#
# A field of type `Any` is used instead of a `FunctionWrapper` because the latter converts
# its arguments to the declared signature: an in-place gradient written into a view would
# be written into a converted copy, and every dual type the AD backends call with (tags
# included) would have to be enumerated.

"""
    DespecializedFunction{K}(f)

A function of an `OptimizationFunction` stored with its type hidden, created when a
problem with [`SciMLBase.AutoSpecialize`](@ref) or [`SciMLBase.AutoDespecialize`](@ref)
is solved. A `SciMLBase.DespecializedParameters` argument is unwrapped before `f` is
called. `K` is the calling convention, which says what the call returns:

  - `:objective`: `f(x, p) -> value`
  - `:fg`: `fg(G, x, p) -> value`
  - `:fgh`: `fgh(G, H, x, p) -> value`
  - `:inplace`: `g(out, args...)`, writing into `out`; returns `nothing`

Use `SciMLBase.unwrapped_f` to get `f` back.
"""
struct DespecializedFunction{K} <: Function
    f::Any
end

SciMLBase.unwrapped_f(w::DespecializedFunction) = SciMLBase.unwrapped_f(w.f)

@inline function _call_despecialized(f, args...)
    return SciMLBase.invoke_with_despecialized_parameters(f, args)
end

# The value an objective-like call returns, given the optimization variables `x` it was
# called with. The wrapped function is called through a field of type `Any`, so without a
# type on this value everything the solver computes from it is inferred as `Any` too.
# AD backends whose values are not `eltype(x)` add methods for their `x` (see
# OptimizationReverseDiffExt).
function _despecialized_value(val, x)
    return if val isa eltype(x)
        val
    else
        _despecialized_value_error(val, eltype(x))
    end
end

function _despecialized_value_error(val, T)
    error("The objective function does not return eltype(x). Got $(typeof(val)) expected $T. Adapt the objective or use FullSpecialize.")
end

function (w::DespecializedFunction{:objective})(x, args...)
    return _despecialized_value(_call_despecialized(w.f, x, args...), x)
end

function (w::DespecializedFunction{:fg})(G, x, args...)
    return _despecialized_value(_call_despecialized(w.f, G, x, args...), x)
end

function (w::DespecializedFunction{:fgh})(G, H, x, args...)
    return _despecialized_value(_call_despecialized(w.f, G, H, x, args...), x)
end

function (w::DespecializedFunction{:inplace})(args...)
    _call_despecialized(w.f, args...)
    return nothing
end

"""
    DespecializedDataIterator(data)

A data iterator passed as the `p` of an `OptimizationProblem` with
[`SciMLBase.AutoDespecialize`](@ref), with its type hidden. It yields each batch in a
`SciMLBase.DespecializedParameters`, so the `p` of every solver step is erased as well.
`SciMLBase.unwrap_parameters` returns the original iterator.
"""
struct DespecializedDataIterator
    data::Any
end

isa_dataiterator(::DespecializedDataIterator) = true
SciMLBase.unwrap_parameters(d::DespecializedDataIterator) = d.data
Base.length(d::DespecializedDataIterator) = length(d.data)

function Base.iterate(d::DespecializedDataIterator, state...)
    next = iterate(d.data, state...)
    next === nothing && return nothing
    batch, state = next
    return SciMLBase.DespecializedParameters(batch), state
end

"""
    despecializes(f::OptimizationFunction) -> Bool

Whether solving with `f` hides the types of its functions (and, for
[`SciMLBase.AutoDespecialize`](@ref), of `p`), see [`despecialize`](@ref). This is the
case for in-place functions with an `AutoSpecialize` or `AutoDespecialize` marker. Most AD
backends differentiate through the dynamic dispatch of the [`DespecializedFunction`](@ref)s;
the Enzyme extension instead unwraps them behind a function barrier and differentiates the
concrete functions.
"""
function despecializes(f::OptimizationFunction)
    spec = SciMLBase.specialization(f)
    spec === SciMLBase.AutoSpecialize || spec === SciMLBase.AutoDespecialize || return false
    return SciMLBase.isinplace(f)
end
despecializes(f) = false

_despecialize_function(::Val, ::Nothing) = nothing
_despecialize_function(::Val, f::DespecializedFunction) = f
_despecialize_function(::Val{K}, f) where {K} = DespecializedFunction{K}(f)

_despecialize_parameters(p::Union{Nothing, SciMLBase.NullParameters}) = p
_despecialize_parameters(p::Union{SciMLBase.DespecializedParameters, DespecializedDataIterator}) = p
function _despecialize_parameters(p)
    return isa_dataiterator(p) ? DespecializedDataIterator(p) :
        SciMLBase.DespecializedParameters(p)
end

"""
    despecialize(f::OptimizationFunction, p) -> (f, p)

Hide the types of the functions of `f` behind [`DespecializedFunction`](@ref)s and, for
[`SciMLBase.AutoDespecialize`](@ref), of `p` behind a `SciMLBase.DespecializedParameters`
(or a [`DespecializedDataIterator`](@ref) for a data iterator). Returns `f` and `p`
unchanged when [`despecializes`](@ref) is false. Cache constructors call this before
`instantiate_function`, so that the AD preparations are built for the erased types.
"""
function despecialize(f::OptimizationFunction{iip}, p) where {iip}
    despecializes(f) || return f, p
    spec = SciMLBase.specialization(f)
    inplace = Val(:inplace)
    f = OptimizationFunction{iip, spec}(
        _despecialize_function(Val(:objective), f.f), f.adtype;
        grad = _despecialize_function(inplace, f.grad),
        fg = _despecialize_function(Val(:fg), f.fg),
        hess = _despecialize_function(inplace, f.hess),
        fgh = _despecialize_function(Val(:fgh), f.fgh),
        hv = _despecialize_function(inplace, f.hv),
        cons = _despecialize_function(inplace, f.cons),
        cons_j = _despecialize_function(inplace, f.cons_j),
        cons_jvp = _despecialize_function(inplace, f.cons_jvp),
        cons_vjp = _despecialize_function(inplace, f.cons_vjp),
        cons_h = _despecialize_function(inplace, f.cons_h),
        lag_h = _despecialize_function(inplace, f.lag_h),
        f.hess_prototype, f.cons_jac_prototype, f.cons_hess_prototype, f.observed,
        f.expr, f.cons_expr, f.sys, f.lag_hess_prototype, f.hess_colorvec,
        f.cons_jac_colorvec, f.cons_hess_colorvec, f.lag_hess_colorvec,
        f.initialization_data
    )
    spec === SciMLBase.AutoDespecialize && (p = _despecialize_parameters(p))
    return f, p
end
despecialize(f, p) = f, p
