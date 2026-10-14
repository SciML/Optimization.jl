# Type erasure for `AutoSpecialize`/`AutoDespecialize` problems: the functions (and, for
# `AutoDespecialize`, `p`) are hidden behind a field of type `Any` before
# `instantiate_function`, so the AD preparations and solver caches have the same types for
# every objective, at the cost of one dynamic dispatch per call. Not a `FunctionWrapper`,
# which would convert arguments (breaking in-place writes into views and dual types).

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
    return SciMLBase.invoke_with_despecialized_parameters(f, map(_unwrap_iterator, args))
end

# Type-asserts the value of a dynamic call so that the solver code infers.
# AD backends whose values are not `eltype(x)` add methods (see OptimizationReverseDiffExt).
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

# Lets `reinit!(cache; p)` assign a new iterator to the despecialized `p` of a cache.
Base.convert(::Type{DespecializedDataIterator}, data) = DespecializedDataIterator(data)
Base.convert(::Type{DespecializedDataIterator}, data::DespecializedDataIterator) = data

isa_dataiterator(::DespecializedDataIterator) = true

# Some solvers pass the whole iterator to a function (AugLag gives it to the constraints),
# so the wrappers unwrap it like the batches.
_unwrap_iterator(d::DespecializedDataIterator) = d.data
_unwrap_iterator(x) = x
SciMLBase.unwrap_parameters(d::DespecializedDataIterator) = d.data
Base.length(d::DespecializedDataIterator) = length(d.data)

function Base.iterate(d::DespecializedDataIterator, state...)
    next = iterate(d.data, state...)
    next === nothing && return nothing
    batch, state = next
    return SciMLBase.DespecializedParameters(batch), state
end

"""
    DespecializedCallback(callback)

The user callback of a problem with [`SciMLBase.AutoDespecialize`](@ref). The solvers
pass the callback an [`OptimizationState`](@ref) holding the despecialized `p`; this
wrapper replaces it by the parameters the problem was given (or the current batch of a
data iterator) before calling `callback`. The call dispatches at runtime on the type of
the unwrapped parameters.
"""
struct DespecializedCallback{C}
    callback::C
end

function (c::DespecializedCallback)(state::OptimizationState, args...)
    unwrapped = OptimizationState(
        state.iter, state.u, state.objective, state.grad, state.hess, state.original,
        SciMLBase.unwrap_parameters(state.p)
    )
    return c.callback(unwrapped, args...)
end
(c::DespecializedCallback)(args...) = c.callback(args...)

"""
    despecialize_callback(f, callback)

`callback` wrapped in a [`DespecializedCallback`](@ref) when `f` has the
[`SciMLBase.AutoDespecialize`](@ref) marker and is despecialized (see
[`despecializes`](@ref)), otherwise `callback` itself. The default callback and `nothing`
are never wrapped, since solvers compare against them.
"""
function despecialize_callback(f::OptimizationFunction, callback)
    despecializes(f) && SciMLBase.specialization(f) === SciMLBase.AutoDespecialize ||
        return callback
    return DespecializedCallback(callback)
end
despecialize_callback(f, callback) = callback
despecialize_callback(f::OptimizationFunction, callback::NullCallback) = callback
despecialize_callback(f::OptimizationFunction, callback::Nothing) = callback
despecialize_callback(f::OptimizationFunction, callback::DespecializedCallback) = callback

"""
    despecializes(f::OptimizationFunction) -> Bool

Whether solving with `f` hides the types of its functions (and, for
[`SciMLBase.AutoDespecialize`](@ref), of `p`), see [`despecialize`](@ref). This is the
case for in-place functions with an `AutoSpecialize` or `AutoDespecialize` marker. Most AD
backends differentiate through the dynamic dispatch of the [`DespecializedFunction`](@ref)s;
the Enzyme extension instead unwraps them behind a function barrier and differentiates the
concrete functions. Symbolic backends (`AutoSymbolics`) and Enzyme inside a `SecondOrder`
or an `AutoSparse` need the concrete functions to instantiate, so problems using them keep
their full specialization.
"""
function despecializes(f::OptimizationFunction)
    spec = SciMLBase.specialization(f)
    spec === SciMLBase.AutoSpecialize || spec === SciMLBase.AutoDespecialize || return false
    return SciMLBase.isinplace(f) && _despecializes_with(f.adtype)
end

# AD choices that need the concrete functions keep full specialization: symbolic backends
# trace them, and Enzyme through DifferentiationInterface cannot see through the dynamic
# dispatch (a plain `AutoEnzyme` unwraps them in the Enzyme extension).
_despecializes_with(::ADTypes.AbstractADType) = true
_despecializes_with(::AutoSymbolics) = false
function _despecializes_with(ad::AutoSparse)
    dense = ADTypes.dense_ad(ad)
    return !(dense isa AutoEnzyme) && _despecializes_with(dense)
end
function _despecializes_with(ad::DifferentiationInterface.SecondOrder)
    return !(ad.inner isa AutoEnzyme || ad.outer isa AutoEnzyme) &&
        _despecializes_with(ad.inner) && _despecializes_with(ad.outer)
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
