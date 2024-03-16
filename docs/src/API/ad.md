# [Automatic Differentiation Construction Choice Recommendations](ad)

The choices for the auto-AD fill-ins with quick descriptions are:

  - `AutoForwardDiff()`: The fastest choice for small optimizations
  - `AutoReverseDiff(compile=false)`: A fast choice for large scalar optimizations
  - `AutoTracker()`: Like ReverseDiff but GPU-compatible
  - `AutoZygote()`: The fastest choice for non-mutating array-based (BLAS) functions
  - `AutoFiniteDiff()`: Finite differencing, not optimal but always applicable
  - `AutoModelingToolkit()`: The fastest choice for large scalar optimizations
  - `AutoEnzyme()`: Highly performant AD choice for type stable and optimized code

## Automatic Differentiation Choice API

The following sections describe the Auto-AD choices in detail.

```@docs
Optimization.AutoForwardDiff
Optimization.AutoFiniteDiff
Optimization.AutoReverseDiff
Optimization.AutoZygote
Optimization.AutoTracker
Optimization.AutoModelingToolkit
Optimization.AutoEnzyme
```
