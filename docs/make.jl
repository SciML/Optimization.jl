using Documenter, Optimization
using FiniteDiff, ForwardDiff, ModelingToolkit, ReverseDiff, Tracker, Zygote

include("pages.jl")

makedocs(sitename = "Optimization.jl",
         authors = "Chris Rackauckas, Vaibhav Kumar Dixit et al.",
         clean = true,
         doctest = false,
         modules = [Optimization, Optimization.SciMLBase, FiniteDiff,
             ForwardDiff, ModelingToolkit, ReverseDiff, Tracker, Zygote],
         format = Documenter.HTML(analytics = "UA-90474609-3",
                                  assets = ["assets/favicon.ico"],
                                  canonical = "https://Optimization.sciml.ai/stable/"),
         pages = pages)

deploydocs(repo = "github.com/SciML/Optimization.jl";
           push_preview = true)
