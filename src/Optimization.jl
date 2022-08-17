"""
$(DocStringExtensions.README)
"""
module Optimization

using DocStringExtensions
using Reexport
@reexport using SciMLBase
using Requires
using DiffResults
using Logging, ProgressLogging, ConsoleProgressMonitor, TerminalLoggers, LoggingExtras
using ArrayInterfaceCore, Base.Iterators, SparseArrays
using Pkg

import SciMLBase: OptimizationProblem, OptimizationFunction, AbstractADType

@enum ObjSense MinSense MaxSense

include("utils.jl")
include("function/function.jl")

function __init__()
    # AD backends
    @require FiniteDiff="6a86dc24-6348-571c-b903-95158fe2bd41" include("function/finitediff.jl")
    @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" include("function/forwarddiff.jl")
    @require ReverseDiff="37e2e3b7-166d-5795-8a7a-e32c996b4267" include("function/reversediff.jl")
    @require Tracker="9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" include("function/tracker.jl")
    @require Zygote="e88e6eb3-aa80-5325-afca-941959d7151f" include("function/zygote.jl")
    @require ModelingToolkit="961ee093-0014-501f-94e3-6117800e7a78" include("function/mtk.jl")
end

export solve

end # module
