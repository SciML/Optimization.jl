module GalacticOptim

using DiffEqBase  
using DiffResults, ForwardDiff
using Optim

include("problem.jl")
include("solve.jl")
include("function.jl")

export OptimizationProblem, OptimizationFunction
export solve

end # module
