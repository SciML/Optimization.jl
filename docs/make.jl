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
        "Tutorials" => [
            "Introduction to GalacticOptim.jl" => "intro.md"
        ],
        "Basics" => [
            "OptimizationProblem" => "problem.md",
            "Solver Options" => "solve.md"
        ],
        "Local Optimizers" => [
            "local_gradient.md",
            "local_derivative_free.md",
            "local_hessian.md",
            "local_hessian_free.md",
        ],
        "Global Optimizers" => [
            "global.md",
            "global_constrained.md"
        ]
    ]
)

deploydocs(
    repo="github.com/SciML/GalacticOptim.jl";
    push_preview=true
)
