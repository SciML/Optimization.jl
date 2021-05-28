decompose_trace(opt::BlackBoxOptim.OptRunController) = BlackBoxOptim.best_candidate(opt)

export BBO

struct BBO
    method::Symbol
    BBO(method) = new(method)
end

BBO() = BBO(:adaptive_de_rand_1_bin_radiuslimited) # the recommended optimizer as default

function __solve(prob::OptimizationProblem, opt::BBO, data = DEFAULT_DATA;
                 cb = (args...) -> (false), maxiters = nothing,
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

    if !(isnothing(maxiters)) && maxiters <= 0.0
        error("The number of maxiters has to be a non-negative and non-zero number.")
    elseif !(isnothing(maxiters))
        maxiters = convert(Int, maxiters)
    end

    _loss = function(θ)
        x = prob.f(θ, prob.p, cur...)
        return first(x)
    end

    bboptre = !(isnothing(maxiters)) ? BlackBoxOptim.bboptimize(_loss;Method = opt.method, SearchRange = [(prob.lb[i], prob.ub[i]) for i in 1:length(prob.lb)], MaxSteps = maxiters, CallbackFunction = _cb, CallbackInterval = 0.0, kwargs...) : BlackBoxOptim.bboptimize(_loss;Method = opt.method, SearchRange = [(prob.lb[i], prob.ub[i]) for i in 1:length(prob.lb)], CallbackFunction = _cb, CallbackInterval = 0.0, kwargs...)

    SciMLBase.build_solution(prob, opt, BlackBoxOptim.best_candidate(bboptre),
                             BlackBoxOptim.best_fitness(bboptre); original=bboptre)
end
