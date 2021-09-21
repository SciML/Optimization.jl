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
            "Basic usage" => "tutorials/intro.md",
            "Rosenbrock function" => "tutorials/rosenbrock.md",
            "Minibatch" => "tutorials/minibatch.md",
            "Symbolic Modeling" => "tutorials/symbolic.md"
        ],

        "API" => [
            "OptimizationProblem" => "API/optimization_problem.md",
            "OptimizationFunction" => "API/optimization_function.md",
            "solve" => "API/solve.md",
            "ModelingToolkit Integration" => "API/modelingtoolkit.md"
        ],
        "Optimizer Packages" => [
            "BlackBoxOptim.jl" => "optimization_packages/blackboxoptim.md",
            "CMAEvolutionStrategy.jl" => "optimization_packages/cmaevolutionstrategy.md",
            "Evolutionary.jl" => "optimization_packages/evolutionary.md",
            "Flux.jl" => "optimization_packages/flux.md",
            "MathOptInterface.jl" => "optimization_packages/mathoptinterface.md",
            "MultistartOptimization.jl" => "optimization_packages/multistartoptimization.md",
            "NLopt.jl" => "optimization_packages/nlopt.md",
            "Nonconvex.jl" => "optimization_packages/nonconvex.md",
            "Optim.jl" => "optimization_packages/optim.md",
            "QuadDIRECT.jl" => "optimization_packages/quaddirect.md"
        ],

        "Local Optimizers" => [
            "local_optimizers/local_gradient.md",
            "local_optimizers/local_derivative_free.md",
            "local_optimizers/local_hessian.md",
            "local_optimizers/local_hessian_free.md"
        ],

        "Global Optimizers" => [
            "global_optimizers/global.md",
            "global_optimizers/global_constrained.md"
        ]
    ]
)

deploydocs(
    repo="github.com/SciML/GalacticOptim.jl";
    push_preview=true
)
