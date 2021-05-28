"""
$(DocStringExtensions.README)
"""
module GalacticOptim

using DocStringExtensions
using Reexport
@reexport using SciMLBase
using Requires
using DiffResults, ForwardDiff, Zygote, ReverseDiff, Tracker, FiniteDiff
using Logging, ProgressLogging, Printf, ConsoleProgressMonitor, TerminalLoggers, LoggingExtras
using ArrayInterface, Base.Iterators

using ForwardDiff: DEFAULT_CHUNK_THRESHOLD
import SciMLBase: OptimizationProblem, OptimizationFunction, AbstractADType, __solve

include("solve/solve.jl")
include("function.jl")

function __init__()
    # Optimization backends
    @require BlackBoxOptim="a134a8b2-14d6-55f6-9291-3336d3ab0209" include("solve/blackboxoptim.jl")
    @require CMAEvolutionStrategy="8d3b24bd-414e-49e0-94fb-163cc3a3e411" include("solve/cmaevolutionstrategy.jl")
    @require Evolutionary="86b6b26d-c046-49b6-aa0b-5f0f74682bd6" include("solve/evolutionary.jl")
    @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" include("solve/flux.jl")
    @require MultistartOptimization="3933049c-43be-478e-a8bb-6e0f7fd53575" include("solve/multistartoptimization.jl")
    @require NLopt="76087f3c-5699-56af-9a33-bf431cd00edd" include("solve/nlopt.jl")
    @require Optim="429524aa-4258-5aef-a3af-852621145aeb" include("solve/optim.jl")
    @require QuadDIRECT="dae52e8d-d666-5120-a592-9e15c33b8d7a" include("solve/quaddirect.jl")
end

export solve

end # module
