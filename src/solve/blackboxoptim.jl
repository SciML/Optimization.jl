abstract type BBO end

for j = string.(BlackBoxOptim.SingleObjectiveMethodNames)
    eval(Meta.parse("Base.@kwdef struct BBO_"*j*" <: BBO method=:"*j*" end"))
    eval(Meta.parse("export BBO_"*j))
end

decompose_trace(opt::BlackBoxOptim.OptRunController) = BlackBoxOptim.best_candidate(opt)

function __map_optimizer_args(prob::OptimizationProblem, opt::BBO;
    cb=nothing,
    maxiters::Union{Number, Nothing}=nothing,
    maxtime::Union{Number, Nothing}=nothing,
    abstol::Union{Number, Nothing}=nothing, 
    reltol::Union{Number, Nothing}=nothing, 
    kwargs...)

    if !isnothing(reltol)
        @warn "reltol is currently not used by $(opt)"
    end

    mapped_args = (; Method = opt.method,
    SearchRange = [(prob.lb[i], prob.ub[i]) for i in 1:length(prob.lb)])

    if !isnothing(cb)
        mapped_args = (; mapped_args..., CallbackFunction = cb, CallbackInterval = 0.0)
    end

    mapped_args = (; mapped_args..., kwargs...)
  
    if !isnothing(maxiters)
        mapped_args = (; mapped_args..., MaxSteps=maxiters)
    end

    if !isnothing(maxtime)
        mapped_args = (; mapped_args..., MaxTime=maxtime)
    end

    if !isnothing(abstol)
        mapped_args = (; mapped_args..., MinDeltaFitnessTolerance=abstol)
  end
  
    return mapped_args
end

function __solve(prob::OptimizationProblem, opt::BBO, data = DEFAULT_DATA;
                 cb = (args...) -> (false), 
                 maxiters::Union{Number, Nothing} = nothing,
                 maxtime::Union{Number, Nothing} = nothing,
                 abstol::Union{Number, Nothing}=nothing,
                 reltol::Union{Number, Nothing}=nothing,
                 progress = false, kwargs...)

    local x, cur, state

    if data != DEFAULT_DATA
        maxiters = length(data)
    end

    cur, state = iterate(data)

    function _cb(trace)
      cb_call = cb(decompose_trace(trace),x...)
      if !(typeof(cb_call) <: Bool)
        error("The callback should return a boolean `halt` for whether to stop the optimization process.")
      end
      if cb_call == true
        BlackBoxOptim.shutdown_optimizer!(trace) #doesn't work
      end
      cur, state = iterate(data, state)
      cb_call
    end

    maxiters = _check_and_convert_maxiters(maxiters)
    maxtime = _check_and_convert_maxtime(maxtime)


    _loss = function(θ)
        x = prob.f(θ, prob.p, cur...)
        return first(x)
    end

    opt_args = _map_optimizer_args(prob, opt, cb=_cb, maxiters=maxiters, maxtime=maxtime,abstol=abstol, reltol=reltol; kwargs...)

    opt_setup = BlackBoxOptim.bbsetup(_loss; opt_args...)

    t0 = time()
    opt_res = BlackBoxOptim.bboptimize(opt_setup)
    t1 = time()
    
    opt_ret = Symbol(opt_res.stop_reason)

    SciMLBase.build_solution(prob, opt, BlackBoxOptim.best_candidate(opt_res),
                             BlackBoxOptim.best_fitness(opt_res); original=opt_res, retcode=opt_ret)
end
