module OptimizationOptimisersMLUtilsExt

using MLUtils
using OptimizationOptimisers

OptimizationOptimisers.isa_dataiterator(::MLUtils.DataLoader) = true

end
