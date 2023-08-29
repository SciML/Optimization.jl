"""
$(DocStringExtensions.README)
"""
module Optimization

using DocStringExtensions
using Reexport
@reexport using SciMLBase, ADTypes

if !isdefined(Base, :get_extension)
    using Requires
end

using Logging, ProgressLogging, ConsoleProgressMonitor, TerminalLoggers, LoggingExtras
using ArrayInterface, Base.Iterators, SparseArrays, LinearAlgebra
using Pkg

import SciMLBase: OptimizationProblem, OptimizationFunction, ObjSense,
    MaxSense, MinSense
export ObjSense, MaxSense, MinSense

include("utils.jl")
include("function.jl")
include("adtypes.jl")
include("cache.jl")

@static if !isdefined(Base, :get_extension)
    function __init__()
        # AD backends
        @require FiniteDiff="6a86dc24-6348-571c-b903-95158fe2bd41" begin
            include("../ext/OptimizationFiniteDiffExt.jl")
            using .OptimizationFiniteDiffExt
        end
        @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" begin
            include("../ext/OptimizationForwardDiffExt.jl")
            using .OptimizationForwardDiffExt
        end
        @require ReverseDiff="37e2e3b7-166d-5795-8a7a-e32c996b4267" begin
            include("../ext/OptimizationReverseDiffExt.jl")
            using .OptimizationReverseDiffExt
        end
        @require Tracker="9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" begin
            include("../ext/OptimizationTrackerExt.jl")
            using .OptimizationTrackerExt
        end
        @require Zygote="e88e6eb3-aa80-5325-afca-941959d7151f" begin
            include("../ext/OptimizationZygoteExt.jl")
            using .OptimizationZygoteExt
        end
        @require ModelingToolkit="961ee093-0014-501f-94e3-6117800e7a78" begin
            include("../ext/OptimizationMTKExt.jl")
            using .OptimizationMTKExt
        end
        @require Enzyme="7da242da-08ed-463a-9acd-ee780be4f1d9" begin
            include("../ext/OptimizationEnzymeExt.jl")
            using .OptimizationEnzymeExt
        end
        @require SparseDiffTools="47a9eef4-7e08-11e9-0b38-333d64bd3804" begin
            @require Symbolics="0c5d862f-8b57-4792-8d23-62f2024744c7" begin
                @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" begin
                    include("../ext/OptimizationSparseForwardDiffExt.jl")
                    using .OptimizationSparseForwardDiffExt
                end

                @require FiniteDiff="6a86dc24-6348-571c-b903-95158fe2bd41" begin
                    include("../ext/OptimizationSparseFiniteDiffExt.jl")
                    using .OptimizationSparseFiniteDiffExt
                end
            end
        end
    end
end

export solve, OptimizationCache

end # module
