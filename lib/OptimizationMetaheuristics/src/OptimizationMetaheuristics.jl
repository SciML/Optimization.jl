module OptimizationMetaheuristics

using Reexport, Optimization, Optimization.SciMLBase
@reexport using Metaheuristics

function initial_population!(opt, prob, bounds, f)
    opt_init = deepcopy(opt)
    opt_init.options.iterations = 2
    Metaheuristics.optimize(f, bounds, opt_init)

    pop_size = opt_init.parameters.N
    population_rand = [bounds[1, :] +
                       rand(length(prob.u0)) .* (bounds[2, :] - bounds[1, :])
                       for i in 1:(pop_size - 1)]
    push!(population_rand, prob.u0)
    population_init = [Metaheuristics.create_child(x, f(x)) for x in population_rand]
    prev_status = Metaheuristics.State(Metaheuristics.get_best(population_init),
                                       population_init)
    opt.parameters.N = pop_size
    opt.status = prev_status
    return nothing
end

function __map_optimizer_args!(prob::OptimizationProblem,
                               opt::Metaheuristics.AbstractAlgorithm;
                               callback = nothing,
                               maxiters::Union{Number, Nothing} = nothing,
                               maxtime::Union{Number, Nothing} = nothing,
                               abstol::Union{Number, Nothing} = nothing,
                               reltol::Union{Number, Nothing} = nothing,
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

    if !isnothing(callback)
        @warn "Callback argument is currently not used by $(typeof(opt).super)"
    end

    if !isnothing(maxiters)
        opt.options.iterations = maxiters
    end

    if !isnothing(maxtime)
        opt.options.time_limit = maxtime
    end

    if !isnothing(abstol)
        opt.options.f_tol = abstol
    end

    if !isnothing(reltol)
        @warn "common reltol is currently not used by $(typeof(opt).super)"
    end
    return nothing
end

function SciMLBase.__solve(prob::OptimizationProblem, opt::Metaheuristics.AbstractAlgorithm;
                           callback = nothing,
                           maxiters::Union{Number, Nothing} = nothing,
                           maxtime::Union{Number, Nothing} = nothing,
                           abstol::Union{Number, Nothing} = nothing,
                           reltol::Union{Number, Nothing} = nothing,
                           progress = false,
                           use_initial::Bool = false,
                           kwargs...)
    local x

    maxiters = Optimization._check_and_convert_maxiters(maxiters)
    maxtime = Optimization._check_and_convert_maxtime(maxtime)

    _loss = function (θ)
        x = prob.f(θ, prob.p)
        return first(x)
    end

    if !isnothing(prob.lb) & !isnothing(prob.ub)
        opt_bounds = [prob.lb prob.ub]'
    else
        error("$(opt) requires lower and upper bounds to be defined.")
    end

    if !isnothing(prob.f.cons)
        @warn "Equality constraints are current not passed on by Optimization"
    end

    if !isnothing(prob.lcons)
        @warn "Inequality constraints are current not passed on by Optimization"
    end

    if !isnothing(prob.ucons)
        @warn "Inequality constraints are current not passed on by Optimization"
    end

    __map_optimizer_args!(prob, opt, callback = callback, maxiters = maxiters,
                          maxtime = maxtime, abstol = abstol, reltol = reltol; kwargs...)

    if use_initial
        initial_population!(opt, prob, opt_bounds, _loss)
    end

    t0 = time()
    opt_res = Metaheuristics.optimize(_loss, opt_bounds, opt)
    t1 = time()

    SciMLBase.build_solution(prob, opt, Metaheuristics.minimizer(opt_res),
                             Metaheuristics.minimum(opt_res); original = opt_res)
end

end
