module OptimizationReverseDiffExt

using DifferentiationInterface, ReverseDiff
using OptimizationBase: OptimizationBase

# ReverseDiff values are `TrackedReal{V, D, O}` with an origin `O` that is not `eltype(x)`'s.
function OptimizationBase._despecialized_value(
        val, x::ReverseDiff.TrackedArray{V, D}
    ) where {V, D}
    return if val isa ReverseDiff.TrackedReal{V, D}
        val
    else
        OptimizationBase._despecialized_value_error(val, ReverseDiff.TrackedReal{V, D})
    end
end

end
