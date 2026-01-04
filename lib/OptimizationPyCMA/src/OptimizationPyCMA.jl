module OptimizationPyCMA

using Reexport
@reexport using OptimizationBase
using PythonCall, SciMLBase

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
SciMLBase.has_init(opt::PyCMAOpt) = true
SciMLBase.allowscallback(::PyCMAOpt) = true
SciMLBase.requiresgradient(::PyCMAOpt) = false
SciMLBase.requireshessian(::PyCMAOpt) = false
SciMLBase.requiresconsjac(::PyCMAOpt) = false
SciMLBase.requiresconshess(::PyCMAOpt) = false

# wrapping OptimizationBase.jl args into a python dict as arguments to PyCMA opts
function __map_optimizer_args(
        prob::OptimizationBase.OptimizationCache, opt::PyCMAOpt;
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        PyCMAargs...
    )
    if !isnothing(reltol)
        @warn "common reltol is currently not used by $(opt)"
    end

    # Converting OptimizationBase.jl args to PyCMA opts
    # OptimizationBase.jl kwargs will overwrite PyCMA kwargs supplied to solve()

    mapped_args = Dict{String, Any}()

    # adding PyCMA args
    merge!(mapped_args, Dict(string(k) => v for (k, v) in PyCMAargs))

    # mapping OptimizationBase.jl args
    mapped_args["bounds"] = (prob.lb, prob.ub)

    if !("verbose" ∈ keys(mapped_args))
        mapped_args["verbose"] = -1
    end

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
    if any(k in keys(stop_dict) for k in ["ftarget", "tolfun", "tolx"])
        return ReturnCode.Success
    elseif any(k in keys(stop_dict) for k in ["maxiter", "maxfevals"])
        return ReturnCode.MaxIters
    elseif "timeout" ∈ keys(stop_dict)
        return ReturnCode.MaxTime
    elseif "callback" ∈ keys(stop_dict)
        return ReturnCode.Terminated
    elseif any(
            k in keys(stop_dict)
                for k in [
                    "tolupsigma", "tolconditioncov", "noeffectcoord", "noeffectaxis",
                    "tolxstagnation", "tolflatfitness", "tolfacupx", "tolstagnation",
                ]
        )
        return ReturnCode.Failure
    else
        return ReturnCode.Default
    end
end

function SciMLBase.__solve(cache::OptimizationCache{O}) where {O <: PyCMAOpt}
    local x

    # wrapping the objective function
    _loss = function (θ)
        x = cache.f(θ, cache.p)
        return first(x)
    end

    _cb = function (es)
        opt_state = OptimizationBase.OptimizationState(;
            iter = pyconvert(Int, es.countiter),
            u = pyconvert(Vector{Float64}, es.best.x),
            p = cache.p,
            objective = pyconvert(Float64, es.best.f),
            original = es
        )

        cb_call = cache.callback(opt_state, x...)
        if !(cb_call isa Bool)
            error("The callback should return a boolean `halt` for whether to stop the optimization process.")
        end
        return if cb_call
            es.opts.set(Dict("termination_callback" => es -> true))
        end
    end

    # doing conversions
    maxiters = OptimizationBase._check_and_convert_maxiters(cache.solver_args.maxiters)
    maxtime = OptimizationBase._check_and_convert_maxtime(cache.solver_args.maxtime)

    # converting the OptimizationBase.jl Args to PyCMA format
    opt_args = __map_optimizer_args(
        cache, cache.opt; cache.solver_args...,
        maxiters = maxiters,
        maxtime = maxtime
    )

    # init the CMAopt class
    es = get_cma().CMAEvolutionStrategy(cache.u0, 1, pydict(opt_args))

    # running the optimization
    t0 = time()
    opt_res = es.optimize(_loss, callback = _cb)
    t1 = time()

    # reading the results
    opt_ret_dict = opt_res.stop()
    retcode = __map_pycma_retcode(pyconvert(Dict{String, Any}, opt_ret_dict))

    # logging and returning results of the optimization
    stats = OptimizationBase.OptimizationStats(;
        iterations = pyconvert(Int, es.countiter),
        time = t1 - t0,
        fevals = pyconvert(Int, es.countevals)
    )

    return SciMLBase.build_solution(
        cache, cache.opt,
        pyconvert(Vector{Float64}, opt_res.result.xbest),
        pyconvert(Float64, opt_res.result.fbest); original = opt_res,
        retcode = retcode,
        stats = stats
    )
end

end # module OptimizationPyCMA
