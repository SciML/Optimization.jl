module SimpleOptimizationEnzymeExt

import SimpleOptimization
import SimpleOptimization.ADTypes: AutoEnzyme

using Enzyme: autodiff_deferred, Reverse, Active

#inlining helps GPU compilation
@inline function SimpleOptimization.instantiate_gradient(f, ::AutoEnzyme)
    return θ -> autodiff_deferred(Reverse, f, Active, Active(θ))[1][1]
end

end
