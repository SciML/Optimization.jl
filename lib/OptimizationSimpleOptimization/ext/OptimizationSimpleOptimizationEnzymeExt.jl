module OptimizationSimpleOptimizationEnzymeExt

import OptimizationSimpleOptimization
import OptimizationSimpleOptimization.ADTypes: AutoEnzyme

using Enzyme

#inlining helps GPU compilation
@inline function OptimizationSimpleOptimization.instantiate_gradient(f, ::AutoEnzyme)
    θ -> autodiff_deferred(Reverse, f, Active, Active(θ))[1][1]
end

end
