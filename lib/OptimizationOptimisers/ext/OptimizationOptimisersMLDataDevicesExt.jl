module OptimizationOptimisersMLDataDevicesExt

using MLDataDevices
using OptimizationOptimisers

OptimizationOptimisers.isa_dataiterator(::DeviceIterator) = (@show "dkjht"; true)

end
