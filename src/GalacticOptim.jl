module GalacticOptim

using DiffEqBase, Requires  
using DiffResults, ForwardDiff, Zygote, ReverseDiff, Tracker
using Optim

include("problem.jl")
include("solve.jl")
include("function.jl")

export OptimizationProblem, OptimizationFunction
export solve

export BBO

end # module
