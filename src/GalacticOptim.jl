module GalacticOptim

using DiffEqBase, Optim, ForwardDiff

include("problem.jl")
include("solve.jl")
include("function.jl")

export OptimizationProblem, OptimizationFunction
export solve

end # module
