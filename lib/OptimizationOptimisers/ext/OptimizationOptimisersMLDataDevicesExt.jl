module OptimizationOptimisersMLDataDevicesExt

using MLDataDevices
using OptimizationOptimisers

OptimizationOptimisers.isa_dataiterator(::DeviceIterator) = true

end
