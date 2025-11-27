# Optimization.jl

The Optimization.jl package provides the common interface for defining and solving optimization problems. All optimization solvers are provided through separate wrapper packages that need to be installed independently.

For a list of available solver packages, see the other pages in this section of the documentation.

Some commonly used solver packages include:

- [OptimizationLBFGSB.jl](@ref lbfgsb) - L-BFGS-B quasi-Newton method with box constraints
- [OptimizationOptimJL.jl](@ref optim) - Wrappers for Optim.jl solvers
- [OptimizationMOI.jl](@ref mathoptinterface) - MathOptInterface solvers
- [OptimizationSophia.jl](@ref sophia) - Sophia optimizer for neural network training

For examples of using these solvers, please refer to their respective documentation pages.
