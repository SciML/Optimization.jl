module SimpleOptimizationEnzymeExt

import SimpleOptimization
import SimpleOptimization.ADTypes: AutoEnzyme

using Enzyme

#inlining helps GPU compilation
@inline function SimpleOptimization.instantiate_gradient(f, ::AutoEnzyme)
    return θ -> autodiff_deferred(Reverse, f, Active, Active(θ))[1][1]
end

end
