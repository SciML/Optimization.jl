module OptimizationSimpleOptimizationForwardDiffExt

import OptimizationSimpleOptimization
import OptimizationSimpleOptimization.ADTypes: AutoForwardDiff

using ForwardDiff

#inlining helps GPU compilation
@inline function OptimizationSimpleOptimization.instantiate_gradient(f, ::AutoForwardDiff)
    θ -> ForwardDiff.gradient(f, θ)
end

end
