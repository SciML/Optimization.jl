using Documenter, Optimization
using FiniteDiff, ForwardDiff, ModelingToolkit, ReverseDiff, Tracker, Zygote

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

include("pages.jl")

makedocs(sitename = "Optimization.jl",
    authors = "Chris Rackauckas, Vaibhav Kumar Dixit et al.",
    modules = [Optimization, Optimization.SciMLBase, FiniteDiff,
        ForwardDiff, ModelingToolkit, ReverseDiff, Tracker, Zygote],
    clean = true, doctest = false,
    strict = [
        :doctest,
        :linkcheck,
        :parse_error,
        :example_block,
        # Other available options are
        # :autodocs_block, :cross_references, :docs_block, :eval_block, :example_block, :footnote, :meta_block, :missing_docs, :setup_block
    ],
    format = Documenter.HTML(analytics = "UA-90474609-3",
        assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/Optimization/stable/"),
    pages = pages)

deploydocs(repo = "github.com/SciML/Optimization.jl";
    push_preview = true)
