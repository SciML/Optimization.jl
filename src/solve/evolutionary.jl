decompose_trace(trace::Evolutionary.OptimizationTrace) = last(trace)

function Evolutionary.trace!(record::Dict{String,Any}, objfun, state, population, method::Evolutionary.AbstractOptimizer, options)
    record["x"] = population
end

function __solve(prob::OptimizationProblem, opt::Evolutionary.AbstractOptimizer, data = DEFAULT_DATA;
                 cb = (args...) -> (false), maxiters = nothing,
                 progress = false, kwargs...)
    local x, cur, state

    if data != DEFAULT_DATA
        maxiters = length(data)
    end

    cur, state = iterate(data)

    function _cb(trace)
        cb_call = cb(decompose_trace(trace).metadata["x"],trace.value...)
        if !(typeof(cb_call) <: Bool)
            error("The callback should return a boolean `halt` for whether to stop the optimization process.")
        end
        cur, state = iterate(data, state)
        cb_call
    end

    if !(isnothing(maxiters)) && maxiters <= 0.0
        error("The number of maxiters has to be a non-negative and non-zero number.")
    elseif !(isnothing(maxiters))
        maxiters = convert(Int, maxiters)
    end

    _loss = function(θ)
        x = prob.f(θ, prob.p, cur...)
        return first(x)
    end

    t0 = time()

    result = Evolutionary.optimize(_loss, prob.u0, opt, !isnothing(maxiters) ? Evolutionary.Options(;iterations = maxiters, callback = _cb, kwargs...)
                                                                        : Evolutionary.Options(;callback = _cb, kwargs...))
    t1 = time()

    SciMLBase.build_solution(prob, opt, Evolutionary.minimizer(result), Evolutionary.minimum(result); original=result)
end
