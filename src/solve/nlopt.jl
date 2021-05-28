function __solve(prob::OptimizationProblem, opt::NLopt.Opt;
                 maxiters = nothing, nstart = 1,
                 local_method = nothing,
                 progress = false, kwargs...)
    local x

    if !(isnothing(maxiters)) && maxiters <= 0.0
        error("The number of maxiters has to be a non-negative and non-zero number.")
    elseif !(isnothing(maxiters))
        maxiters = convert(Int, maxiters)
    end

    f = instantiate_function(prob.f,prob.u0,prob.f.adtype,prob.p)

    _loss = function(θ)
        x = f.f(θ, prob.p)
        return x[1]
    end

    fg! = function (θ,G)
        if length(G) > 0
            f.grad(G, θ)
        end

        return _loss(θ)
    end

    NLopt.min_objective!(opt, fg!)

    if prob.ub !== nothing
        NLopt.upper_bounds!(opt, prob.ub)
    end
    if prob.lb !== nothing
        NLopt.lower_bounds!(opt, prob.lb)
    end
    if !(isnothing(maxiters))
        NLopt.maxeval!(opt, maxiters)
    end
    if nstart > 1 && local_method !== nothing
        NLopt.local_optimizer!(opt, local_method)
        if !(isnothing(maxiters))
            NLopt.maxeval!(opt, nstart * maxiters)
        end
    end

    t0 = time()
    (minf,minx,ret) = NLopt.optimize(opt, prob.u0)
    _time = time()

    SciMLBase.build_solution(prob, opt, minx, minf; original=nothing)
end
