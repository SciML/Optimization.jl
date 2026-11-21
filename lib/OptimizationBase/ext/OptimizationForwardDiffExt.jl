module OptimizationForwardDiffExt

using ForwardDiff: ForwardDiff
using OptimizationBase: OptimizationBase

@inline function OptimizationBase._cons_out_eltype(
        ::Type{T}, ::Type{P}
    ) where {T, P <: ForwardDiff.Dual}
    return promote_type(T, P)
end

end
