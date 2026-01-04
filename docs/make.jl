using Documenter, Optimization
using OptimizationLBFGSB, OptimizationSophia

cp(joinpath(@__DIR__, "Manifest.toml"), joinpath(@__DIR__, "src/assets/Manifest.toml"), force = true)
cp(joinpath(@__DIR__, "Project.toml"), joinpath(@__DIR__, "src/assets/Project.toml"), force = true)

include("pages.jl")

makedocs(
    sitename = "Optimization.jl",
    authors = "Chris Rackauckas, Vaibhav Kumar Dixit et al.",
    modules = [
        Optimization, Optimization.SciMLBase, Optimization.OptimizationBase, Optimization.ADTypes,
        OptimizationLBFGSB, OptimizationSophia,
    ],
    clean = true, doctest = false, linkcheck = true,
    warnonly = [:missing_docs, :cross_references],
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
