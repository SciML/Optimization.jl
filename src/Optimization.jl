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

@static if !isdefined(Base, :get_extension)
    function __init__()
        # AD backends
        @require FiniteDiff="6a86dc24-6348-571c-b903-95158fe2bd41" begin
            include("../ext/OptimizationFinitediffExt.jl")
            using .OptimizationFinitediffExt
        end
        @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" begin
            include("../ext/OptimizationForwarddiffExt.jl")
            using .OptimizationForwarddiffExt
        end
        @require ReverseDiff="37e2e3b7-166d-5795-8a7a-e32c996b4267" begin
            include("../ext/OptimizationReversediffExt.jl")
            using .OptimizationReversediffExt
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
    end
end

function Base.getproperty(cache::SciMLBase.AbstractOptimizationCache, x::Symbol)
    if x in fieldnames(Optimization.ReInitCache)
        return getfield(cache.reinit_cache, x)
    end
    return getfield(cache, x)
end

SciMLBase.has_reinit(cache::SciMLBase.AbstractOptimizationCache) = true
function SciMLBase.reinit!(cache::SciMLBase.AbstractOptimizationCache; p = missing,
                           u0 = missing)
    if p === missing && u0 === missing
        p, u0 = cache.p, cache.u0
    else # at least one of them has a value
        if p === missing
            p = cache.p
        end
        if u0 === missing
            u0 = cache.u0
        end
        if (eltype(p) <: Pair && !isempty(p)) || (eltype(u0) <: Pair && !isempty(u0)) # one is a non-empty symbolic map
            hasproperty(cache.f, :sys) && hasfield(typeof(cache.f.sys), :ps) ||
                throw(ArgumentError("This cache does not support symbolic maps with `remake`, i.e. it does not have a symbolic origin." *
                                    " Please use `remake` with the `p` keyword argument as a vector of values, paying attention to parameter order."))
            hasproperty(cache.f, :sys) && hasfield(typeof(cache.f.sys), :states) ||
                throw(ArgumentError("This cache does not support symbolic maps with `remake`, i.e. it does not have a symbolic origin." *
                                    " Please use `remake` with the `u0` keyword argument as a vector of values, paying attention to state order."))
            p, u0 = SciMLBase.process_p_u0_symbolic(cache, p, u0)
        end
    end

    cache.reinit_cache.p = p
    cache.reinit_cache.u0 = u0

    return cache
end

export solve

end # module
