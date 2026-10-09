# Constraint Jacobian products (`cons_vjp`, `cons_jvp`).
#
# Which products an instantiated function carries is decided by `OptimizationCache` from the
# solver traits: a solver that only *allows* a product (`allowsconsvjp`) has its own fallback
# (typically through `cons_j`), so it is given the product only when that is matrix-free
# (`_matrixfree_cons_vjp`); a solver that *requires* it (`requiresconsvjp`) always gets one,
# built through the full Jacobian when that is the only way. `instantiate_function` itself
# builds a requested product from, in order: the user's product, the user's `cons_j`, the
# backend's native product, or an AD Jacobian.

# The mode of the backend used for first-order derivatives: `SecondOrder` differentiates
# first-order quantities with its inner backend, and `ADTypes.mode` of a `SecondOrder` reports
# the outer one.
_first_order_mode(adtype::ADTypes.AbstractADType) = ADTypes.mode(adtype)
_first_order_mode(adtype::DifferentiationInterface.SecondOrder) = _first_order_mode(adtype.inner)
_first_order_mode(adtype::ADTypes.AutoSparse) = _first_order_mode(adtype.dense_ad)
_first_order_mode(::SciMLBase.NoAD) = nothing

# Whether the backend applies `Jᵀ` (resp. `J`) to a vector without materializing `J`. DI has
# no reverse pass for forward-mode backends and synthesizes a pullback from `length(x)`
# pushforwards (and a pushforward from `num_cons` pullbacks for reverse-mode ones), which is
# more expensive than one chunked/colored Jacobian.
function _native_vjp(adtype)
    return _first_order_mode(adtype) isa
        Union{ADTypes.ReverseMode, ADTypes.ForwardOrReverseMode, ADTypes.SymbolicMode}
end
function _native_jvp(adtype)
    return _first_order_mode(adtype) isa
        Union{ADTypes.ForwardMode, ADTypes.ForwardOrReverseMode, ADTypes.SymbolicMode}
end

# Whether `instantiate_function(f, x, adtype, ...; cons_vjp = true)` (resp. `cons_jvp`) can
# provide the product without materializing the constraint Jacobian: the user supplied it, or
# the backend has a native mode for it and the user did not supply a `cons_j` (which takes
# precedence over differentiating `f.cons`).
# Solvers that only *allow* the product should request it only when this holds.
function _matrixfree_cons_vjp(f, adtype)
    return f.cons_vjp !== nothing ||
        (f.cons !== nothing && f.cons_j === nothing && _native_vjp(adtype))
end
function _matrixfree_cons_jvp(f, adtype)
    return f.cons_jvp !== nothing ||
        (f.cons !== nothing && f.cons_j === nothing && _native_jvp(adtype))
end

# The `cons_vjp`/`cons_jvp` flags `OptimizationCache` passes to `instantiate_function`.
function _request_cons_vjp(opt, f)
    return SciMLBase.requiresconsvjp(opt) ||
        (SciMLBase.allowsconsvjp(opt) && _matrixfree_cons_vjp(f, f.adtype))
end
function _request_cons_jvp(opt, f)
    return SciMLBase.requiresconsjvp(opt) ||
        (SciMLBase.allowsconsjvp(opt) && _matrixfree_cons_jvp(f, f.adtype))
end

# A `θ` with a wider eltype than the buffer (e.g. duals from an outer AD pass) cannot be written
# into it, so widen per call; `similar` keeps a sparse buffer's structure. The eltypes are
# known at compile time, so this is type stable.
@inline function _jac_buffer(Jbuf, θ)
    T = promote_type(eltype(θ), eltype(Jbuf))
    return T === eltype(Jbuf) ? Jbuf : similar(Jbuf, T)
end

# In-place `res = Jᵀv` / `res = Jv` through a materialized Jacobian, `jac!(J, θ)` filling `J`.
function _cons_vjp_through_jacobian(jac!, Jbuf)
    return let jac! = jac!, Jbuf = Jbuf
        function (res, θ, v)
            J = _jac_buffer(Jbuf, θ)
            jac!(J, θ)
            return mul!(res, transpose(J), v)
        end
    end
end
function _cons_jvp_through_jacobian(jac!, Jbuf)
    return let jac! = jac!, Jbuf = Jbuf
        function (res, θ, v)
            J = _jac_buffer(Jbuf, θ)
            jac!(J, θ)
            return mul!(res, J, v)
        end
    end
end

# Buffer for a user-supplied in-place `cons_j`, which may rely on the structure of
# `cons_jac_prototype` (e.g. write `nonzeros(J)` directly).
function _user_cons_jac_buffer(f, x, num_cons)
    f.cons_jac_prototype === nothing && return zeros(eltype(x), num_cons, length(x))
    J = similar(f.cons_jac_prototype, eltype(x))
    J isa SparseMatrixCSC ? fill!(nonzeros(J), zero(eltype(J))) : fill!(J, zero(eltype(J)))
    return J
end

# Products built from a user-supplied `cons_j`.
function _user_cons_vjp(f::OptimizationFunction{true}, x, p, num_cons)
    jac! = let f = f, p = p
        (J, θ) -> f.cons_j(J, θ, p)
    end
    return _cons_vjp_through_jacobian(jac!, _user_cons_jac_buffer(f, x, num_cons))
end
function _user_cons_jvp(f::OptimizationFunction{true}, x, p, num_cons)
    jac! = let f = f, p = p
        (J, θ) -> f.cons_j(J, θ, p)
    end
    return _cons_jvp_through_jacobian(jac!, _user_cons_jac_buffer(f, x, num_cons))
end
function _user_cons_vjp(f::OptimizationFunction{false}, x, p, num_cons)
    return let f = f, p = p
        (θ, v) -> transpose(f.cons_j(θ, p)) * v
    end
end
function _user_cons_jvp(f::OptimizationFunction{false}, x, p, num_cons)
    return let f = f, p = p
        (θ, v) -> f.cons_j(θ, p) * v
    end
end
