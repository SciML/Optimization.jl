function __solve(prob::OptimizationProblem, opt::MultistartOptimization.TikTak;
                 local_method, local_maxiters = nothing,
                 progress = false, kwargs...)
    local x, _loss

    if !(isnothing(local_maxiters)) && local_maxiters <= 0.0
        error("The number of local_maxiters has to be a non-negative and non-zero number.")
    else !(isnothing(local_maxiters))
        local_maxiters = convert(Int, local_maxiters)
    end

    _loss = function(θ)
        x = prob.f(θ, prob.p)
        return first(x)
    end

    t0 = time()

    P = MultistartOptimization.MinimizationProblem(_loss, prob.lb, prob.ub)
    multistart_method = opt
    if !(isnothing(local_maxiters))
        local_method = MultistartOptimization.NLoptLocalMethod(local_method, maxeval = local_maxiters)
    else
        local_method = MultistartOptimization.NLoptLocalMethod(local_method)
    end
    p = MultistartOptimization.multistart_minimization(multistart_method, local_method, P)

    t1 = time()

    SciMLBase.build_solution(prob, opt, p.location, p.value; original=p)
end
