module OptimizationReverseDiffExt

using DifferentiationInterface, ReverseDiff
using OptimizationBase: OptimizationBase

# ReverseDiff calls the objective with a `TrackedArray{V, D}` and the value is a
# `TrackedReal{V, D, O}`, not `eltype(x)`: its origin `O` is `Nothing` for a computed value
# and the array when the objective returns an element of `x`. Only ReverseDiff's seeding
# code sees this value, so the open `O` costs no inference in the solver.
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
