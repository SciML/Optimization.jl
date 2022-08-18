module OptimizationEvolutionary

using Reexport, Optimization, Optimization.SciMLBase
@reexport using Evolutionary

decompose_trace(trace::Evolutionary.OptimizationTrace) = last(trace)
decompose_trace(trace::Evolutionary.OptimizationTraceRecord) = trace

function Evolutionary.trace!(record::Dict{String, Any}, objfun, state, population,
                             method::Evolutionary.AbstractOptimizer, options)
    record["x"] = population
end

function __map_optimizer_args(prob::OptimizationProblem,
                              opt::Evolutionary.AbstractOptimizer;
                              callback = nothing,
                              maxiters::Union{Number, Nothing} = nothing,
                              maxtime::Union{Number, Nothing} = nothing,
                              abstol::Union{Number, Nothing} = nothing,
                              reltol::Union{Number, Nothing} = nothing,
                              kwargs...)
    mapped_args = (;)

    mapped_args = (; mapped_args..., kwargs...)

    if !isnothing(callback)
        mapped_args = (; mapped_args..., callback = callback)
    end

    if !isnothing(maxiters)
        mapped_args = (; mapped_args..., iterations = maxiters)
    end

    if !isnothing(maxtime)
        mapped_args = (; mapped_args..., time_limit = maxtime)
    end

    if !isnothing(abstol)
        mapped_args = (; mapped_args..., abstol = abstol)
    end

    if !isnothing(reltol)
        mapped_args = (; mapped_args..., reltol = reltol)
    end

    return Evolutionary.Options(; mapped_args...)
end

function SciMLBase.__solve(prob::OptimizationProblem, opt::Evolutionary.AbstractOptimizer,
                           data = Optimization.DEFAULT_DATA;
                           callback = (args...) -> (false),
                           maxiters::Union{Number, Nothing} = nothing,
                           maxtime::Union{Number, Nothing} = nothing,
                           abstol::Union{Number, Nothing} = nothing,
                           reltol::Union{Number, Nothing} = nothing,
                           progress = false, kwargs...)
    local x, cur, state

    if data != Optimization.DEFAULT_DATA
        maxiters = length(data)
    end

    cur, state = iterate(data)

    function _cb(trace)
        cb_call = callback(decompose_trace(trace).metadata["x"], trace.value...)
        if !(typeof(cb_call) <: Bool)
            error("The callback should return a boolean `halt` for whether to stop the optimization process.")
        end
        cur, state = iterate(data, state)
        cb_call
    end

    maxiters = Optimization._check_and_convert_maxiters(maxiters)
    maxtime = Optimization._check_and_convert_maxtime(maxtime)

    _loss = function (θ)
        x = prob.f(θ, prob.p, cur...)
        return first(x)
    end

    opt_args = __map_optimizer_args(prob, opt, callback = _cb, maxiters = maxiters,
                                    maxtime = maxtime, abstol = abstol, reltol = reltol;
                                    kwargs...)

    t0 = time()
    if isnothing(prob.ub) | isnothing(prob.ub)
        opt_res = Evolutionary.optimize(_loss, prob.u0, opt, opt_args)
    else
        cons = Evolutionary.BoxConstraints(prob.lb, prob.ub)
        opt_res = Evolutionary.optimize(_loss, cons, prob.u0, opt, opt_args)
    end
    t1 = time()
    opt_ret = Symbol(Evolutionary.converged(opt_res))

    SciMLBase.build_solution(prob, opt, Evolutionary.minimizer(opt_res),
                             Evolutionary.minimum(opt_res); original = opt_res,
                             retcode = opt_ret)
end

end
