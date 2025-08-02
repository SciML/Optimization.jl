module OptimizationMLUtilsExt

using MLUtils
using OptimizationBase

OptimizationBase.isa_dataiterator(::MLUtils.DataLoader) = true

end
