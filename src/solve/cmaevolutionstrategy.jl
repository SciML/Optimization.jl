export CMAEvolutionStrategyOpt
struct CMAEvolutionStrategyOpt end

function __solve(prob::OptimizationProblem, opt::CMAEvolutionStrategyOpt, data = DEFAULT_DATA;
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

    _loss = function(θ)
        x = prob.f(θ, prob.p, cur...)
        return first(x)
    end


    if !(isnothing(maxiters)) && maxiters <= 0.0
        error("The number of maxiters has to be a non-negative and non-zero number.")
    elseif !(isnothing(maxiters))
        maxiters = convert(Int, maxiters)
    end

    result = CMAEvolutionStrategy.minimize(_loss, prob.u0, 0.1; lower = prob.lb, upper = prob.ub, maxiter = maxiters, kwargs...)
    CMAEvolutionStrategy.print_header(result)
    CMAEvolutionStrategy.print_result(result)
    println("\n")
    criterion = true

    if (result.stop.reason === :maxtime) #this is an arbitrary choice of convergence (based on the stop.reason values)
        criterion = false
    end

    SciMLBase.build_solution(prob, opt, result.logger.xbest[end], result.logger.fbest[end]; original=result)
end
