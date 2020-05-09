module GalacticOptim

using DiffEqBase, Optim

include("problem.jl")
include("solve.jl")

export OptimizationProblem
export solve

end # module
