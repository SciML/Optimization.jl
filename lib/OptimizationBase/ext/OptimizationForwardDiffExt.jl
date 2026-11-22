module OptimizationForwardDiffExt

using ForwardDiff: ForwardDiff
using OptimizationBase: OptimizationBase

# Dispatch on ForwardDiff's documented scalar carrier preserves partials for dual
# parameters and keeps the state-derived buffer type for ordinary parameters.
@inline function OptimizationBase._lag_cons_out_eltype(
        ::Type{T}, ::Type{P}
    ) where {T, P <: ForwardDiff.Dual}
    return promote_type(T, P)
end

end
