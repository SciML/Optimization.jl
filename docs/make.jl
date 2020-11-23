using Documenter, GalacticOptim

makedocs(
    sitename="GalacticOptim.jl",
    authors="Chris Rackauckas, Vaibhav Kumar Dixit et al.",
    clean=true,
    doctest=false,
    modules=[GalacticOptim],

    format=Documenter.HTML(assets=["assets/favicon.ico"],
                           canonical="https://galacticoptim.sciml.ai/stable/"),

    pages=[
        "GalacticOptim.jl: Unified Global Optimization Package" => "index.md",
        "Usage" => "usage.md",
        "Examples" => "examples.md"
    ]
)

deploydocs(
    repo="github.com/SciML/GalacticOptim.jl";
    push_preview=true
)
