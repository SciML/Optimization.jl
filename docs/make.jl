using Documenter, Optimization
using FiniteDiff, ForwardDiff, ModelingToolkit, ReverseDiff, Tracker, Zygote
using ADTypes

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

include("pages.jl")

makedocs(sitename = "Optimization.jl",
    authors = "Chris Rackauckas, Vaibhav Kumar Dixit et al.",
    modules = [Optimization, Optimization.SciMLBase, Optimization.OptimizationBase,
        FiniteDiff, ForwardDiff, ModelingToolkit, ReverseDiff, Tracker, Zygote, ADTypes],
    clean = true, doctest = false, linkcheck = true,
    warnonly = [:missing_docs, :cross_references],
    format = Documenter.HTML(assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/Optimization/stable/"),
    pages = pages)

deploydocs(repo = "github.com/SciML/Optimization.jl";
    push_preview = true)
