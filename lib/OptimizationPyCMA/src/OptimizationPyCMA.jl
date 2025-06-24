module OptimizationPyCMA

using Reexport
@reexport using Optimization
using PythonCall, Optimization.SciMLBase

export PyCMAOpt

struct PyCMAOpt end

# importing PyCMA
const cma = Ref{Py}()
function get_cma()
    if !isassigned(cma) || cma[] === nothing
        cma[] = pyimport("cma")
    end
    return cma[]
end

# Defining the SciMLBase interface for PyCMAOpt

SciMLBase.allowsbounds(::PyCMAOpt) = true
SciMLBase.supports_opt_cache_interface(opt::PyCMAOpt) = true
SciMLBase.requiresgradient(::PyCMAOpt) = false
SciMLBase.requireshessian(::PyCMAOpt) = false
SciMLBase.requiresconsjac(::PyCMAOpt) = false
SciMLBase.requiresconshess(::PyCMAOpt) = false

# wrapping Optimization.jl args into a python dict as arguments to PyCMA opts
function __map_optimizer_args(prob::OptimizationCache, opt::PyCMAOpt;
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing)
    if !isnothing(reltol)
        @warn "common reltol is currently not used by $(opt)"
    end

    mapped_args = Dict(
        "verbose" => -5,
        "bounds" => (prob.lb, prob.ub),
        )

    if !isnothing(abstol)
        mapped_args["tolfun"] = abstol
    end

    if !isnothing(reltol)
        mapped_args["tolfunrel"] = reltol 
    end

    if !isnothing(maxtime)
        mapped_args["timeout"] = maxtime
    end 

    if !isnothing(maxiters)
        mapped_args["maxiter"] = maxiters
    end

    return mapped_args
end

function __map_pycma_retcode(stop_dict::Dict{String, Any})
    # mapping termination conditions to SciMLBase return codes
    if any(k ∈ keys(stop_dict) for k in ["ftarget", "tolfun", "tolx"])
        return ReturnCode.Success
    elseif any(k ∈ keys(stop_dict) for k in ["maxiter", "maxfevals"])
        return ReturnCode.MaxIters
    elseif "timeout" ∈ keys(stop_dict)
        return ReturnCode.MaxTime
    elseif "callback" ∈ keys(stop_dict)
        return ReturnCode.Terminated
    elseif any(k ∈ keys(stop_dict) for k in ["tolupsigma", "tolconditioncov", "noeffectcoord", "noeffectaxis", "tolxstagnation", "tolflatfitness", "tolfacupx", "tolstagnation"])
        return ReturnCode.Failure
    else
        return ReturnCode.Default
    end
end

function SciMLBase.__solve(cache::OptimizationCache{
        F,
        RC,
        LB,
        UB,
        LC,
        UC,
        S,
        O,
        D,
        P,
        C
}) where {
        F,
        RC,
        LB,
        UB,
        LC,
        UC,
        S,
        O <:
        PyCMAOpt,
        D,
        P,
        C
}
    local x

    # doing conversions
    maxiters = Optimization._check_and_convert_maxiters(cache.solver_args.maxiters)
    maxtime = Optimization._check_and_convert_maxtime(cache.solver_args.maxtime)

    # wrapping the objective function
    _loss = function (θ)
        x = cache.f(θ, cache.p)
        return first(x)
    end

    # converting the Optimization.jl Args to PyCMA format
    opt_args = __map_optimizer_args(cache, cache.opt; cache.solver_args...,
        maxiters = maxiters,
        maxtime = maxtime)
    
    # init the CMAopt class
    es = get_cma().CMAEvolutionStrategy(cache.u0, 1, pydict(opt_args))
    logger = es.logger

    # running the optimization
    t0 = time()
    opt_res = es.optimize(_loss)
    t1 = time()

    # loading logged files from disk
    logger.load()

    # reading the results
    opt_ret_dict = opt_res.stop()
    retcode = __map_pycma_retcode(pyconvert(Dict{String, Any}, opt_ret_dict))
    
    # logging and returning results of the optimization
    stats = Optimization.OptimizationStats(;
        iterations = length(logger.xmean),
        time = t1 - t0,
        fevals = length(logger.xmean))
    
    SciMLBase.build_solution(cache, cache.opt,
        pyconvert(Float64, logger.xrecent[-1][-1]),
        pyconvert(Float64, logger.f[-1][-1]); original = opt_res,
        retcode = retcode,
        stats = stats)
end

end # module OptimizationPyCMA
