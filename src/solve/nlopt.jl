export NLO

struct NLO
    method::Symbol
    NLO(method) = new(method)
end

function __solve(prob::OptimizationProblem, opt::Union{NLO, NLopt.Opt};
                 maxiters = nothing,
                 local_method::Union{NLO, NLopt.Opt, Nothing} = nothing,
                 local_maxiters = nothing,
                 local_options::Union{NamedTuple,Nothing} = nothing,
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

    if isa(opt,NLopt.Opt)
        if ndims(opt) != length(prob.u0)
            error("Passed NLopt.Opt optimization dimension does not match OptimizationProblem dimension.")
        end
        model = opt
    else
        model = NLopt.Opt(opt.method, length(prob.u0))
    end

    NLopt.min_objective!(model, fg!)

    if prob.ub !== nothing
        NLopt.upper_bounds!(model, prob.ub)
    end

    if prob.lb !== nothing
        NLopt.lower_bounds!(model, prob.lb)
    end

    if !(isnothing(maxiters))
        NLopt.maxeval!(model, maxiters)
    end

    if local_method !== nothing
        if isa(local_method,NLopt.Opt)
            if ndims(local_method) != length(prob.u0)
                error("Passed local NLopt.Opt optimization dimension does not match OptimizationProblem dimension.")
            end
        else
            local_method = NLopt.Opt(local_method.method, length(prob.u0))
        end

        if !(isnothing(local_maxiters))
            NLopt.maxeval!(local_method, local_maxiters)
        end

        if !isnothing(local_options)
            for j in Dict(pairs(local_options))
                eval(Meta.parse("NLopt."*string(j.first)*"!"))(local_method, j.second)
            end
        end

        NLopt.local_optimizer!(model, local_method)
    end

    # add optimiser options from kwargs
    for j in kwargs
        eval(Meta.parse("NLopt."*string(j.first)*"!"))(model, j.second)
    end

    # print(model)

    t0 = time()
    (minf,minx,ret) = NLopt.optimize(model, prob.u0)
    _time = time()

    SciMLBase.build_solution(prob, opt, minx, minf; original=model, retcode=ret)
end
