function __map_optimizer_args(prob::OptimizationProblem, opt::Metaheuristics.AbstractAlgorithm;
    cb=nothing,
    maxiters::Union{Number, Nothing}=nothing,
    maxtime::Union{Number, Nothing}=nothing,
    abstol::Union{Number, Nothing}=nothing, 
    reltol::Union{Number, Nothing}=nothing, 
    kwargs...)

    for j in kwargs
        if j.first .∈ Ref(propertynames(Metaheuristics.Information()))
            error("Set $(j.first) by directly passing it to Information Structure which is passed to $(typeof(opt)) algorithms when calling solve().")
        elseif j.first .∈ Ref(propertynames(Metaheuristics.Options()))
            setproperty!(opt.options, j.first, j.second)
        else
            error("$(j.first) keyword is not a valid option for $(typeof(opt).super) algorithm.")
        end
    end

    if !isnothing(cb)
        @warn "Callback argument is currently not used by $(typeof(opt).super)"
    end
  
    if !isnothing(maxiters)
        opt.options.iterations = maxiters
    end

    if !isnothing(maxtime)
        opt.options.time_limit =maxtime
    end

    if !isnothing(abstol)
        opt.options.f_tol=abstol
    end

    if !isnothing(reltol)
        @warn "common reltol is currently not used by $(typeof(opt).super)"
    end
    return nothing
end

function __solve(prob::OptimizationProblem, opt::Metaheuristics.AbstractAlgorithm;
                 cb = nothing, 
                 maxiters::Union{Number, Nothing} = nothing,
                 maxtime::Union{Number, Nothing} = nothing,
                 abstol::Union{Number, Nothing}=nothing,
                 reltol::Union{Number, Nothing}=nothing,
                 progress = false, kwargs...)

    local x

    maxiters = _check_and_convert_maxiters(maxiters)
    maxtime = _check_and_convert_maxtime(maxtime)


    _loss = function(θ)
        x = prob.f(θ, prob.p)
        return first(x)
    end
    
    if !isnothing(prob.lb) & !isnothing(prob.ub)
        opt_bounds = [prob.lb prob.ub]'
    else
        error("$(opt) requires lower and upper bounds to be defined.")
    end

    _map_optimizer_args(prob, opt, cb=cb, maxiters=maxiters, maxtime=maxtime,abstol=abstol, reltol=reltol; kwargs...)

    t0 = time()
    opt_res = Metaheuristics.optimize(_loss, opt_bounds, opt)
    t1 = time()
    
    SciMLBase.build_solution(prob, opt, Metaheuristics.minimizer(opt_res), Metaheuristics.minimum(opt_res); original=opt_res)
end