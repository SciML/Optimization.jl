# [Automatic Differentiation Construction Choice Recommendations](@id ad)

The choices for the auto-AD fill-ins with quick descriptions are:

  - `AutoForwardDiff()`: The fastest choice for small optimizations
  - `AutoReverseDiff(compile=false)`: A fast choice for large scalar optimizations
  - `AutoTracker()`: Like ReverseDiff but GPU-compatible
  - `AutoZygote()`: The fastest choice for non-mutating array-based (BLAS) functions
  - `AutoFiniteDiff()`: Finite differencing, not optimal but always applicable
  - `AutoSymbolics()`: The fastest choice for large scalar optimizations
  - `AutoEnzyme()`: Highly performant AD choice for type stable and optimized code
  - `AutoMooncake()`: Like Zygote and ReverseDiff, but supports GPU and mutating code

## Automatic Differentiation Choice API

The following sections describe the Auto-AD choices in detail. These types are defined in the [ADTypes.jl](https://github.com/SciML/ADTypes.jl) package.

```@docs
ADTypes.AutoForwardDiff
ADTypes.AutoFiniteDiff
ADTypes.AutoReverseDiff
ADTypes.AutoZygote
ADTypes.AutoTracker
ADTypes.AutoSymbolics
ADTypes.AutoEnzyme
ADTypes.AutoMooncake
```
