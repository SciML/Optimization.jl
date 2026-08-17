using Documenter, Optimization
using ADTypes
# The `@docs` entries under API/ name SciMLBase types directly; nothing re-exports
# the module binding any more, so bind it here rather than relying on that leak.
using SciMLBase
using OptimizationAuglag, OptimizationBBO, OptimizationBase
using OptimizationCMAEvolutionStrategy, OptimizationGCMAES, OptimizationIpopt
using OptimizationLBFGSB, OptimizationMadNLP, OptimizationManopt
using OptimizationNLPModels, OptimizationNOMAD, OptimizationODE
using OptimizationPolyalgorithms, OptimizationPRIMA, OptimizationPyCMA
using OptimizationQuadDIRECT, OptimizationSciPy, OptimizationSophia
using OptimizationSpeedMapping, SimpleOptimization

cp(joinpath(@__DIR__, "Manifest.toml"), joinpath(@__DIR__, "src/assets/Manifest.toml"), force = true)
cp(joinpath(@__DIR__, "Project.toml"), joinpath(@__DIR__, "src/assets/Project.toml"), force = true)

include("pages.jl")

makedocs(
    sitename = "Optimization.jl",
    authors = "Chris Rackauckas, Vaibhav Kumar Dixit et al.",
    modules = [
        Optimization, Optimization.OptimizationBase,
        OptimizationAuglag, OptimizationBBO, OptimizationBase,
        OptimizationCMAEvolutionStrategy, OptimizationGCMAES, OptimizationIpopt,
        OptimizationLBFGSB, OptimizationMadNLP, OptimizationManopt,
        OptimizationNLPModels, OptimizationNOMAD, OptimizationODE,
        OptimizationPolyalgorithms, OptimizationPRIMA, OptimizationPyCMA,
        OptimizationQuadDIRECT, OptimizationSciPy, OptimizationSophia,
        OptimizationSpeedMapping, SimpleOptimization,
    ],
    clean = true, linkcheck = true,
    format = Documenter.HTML(
        assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/Optimization/stable/"
    ),
    pages = pages
)

deploydocs(
    repo = "github.com/SciML/Optimization.jl";
    push_preview = true
)
