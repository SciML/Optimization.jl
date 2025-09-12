module OptimizationMLDataDevicesExt

using MLDataDevices
using OptimizationBase

OptimizationBase.isa_dataiterator(::DeviceIterator) = true

end
