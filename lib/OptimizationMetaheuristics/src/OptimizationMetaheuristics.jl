module OptimizationMetaheuristics

using Reexport
@reexport using Metaheuristics, OptimizationBase
using SciMLBase

SciMLBase.requiresbounds(opt::Metaheuristics.AbstractAlgorithm) = true
SciMLBase.allowsbounds(opt::Metaheuristics.AbstractAlgorithm) = true
SciMLBase.allowscallback(opt::Metaheuristics.AbstractAlgorithm) = false
@static if isdefined(SciMLBase, :supports_opt_cache_interface)
    SciMLBase.supports_opt_cache_interface(opt::Metaheuristics.AbstractAlgorithm) = true
end
@static if isdefined(OptimizationBase, :supports_opt_cache_interface)
    function OptimizationBase.supports_opt_cache_interface(opt::Metaheuristics.AbstractAlgorithm)
        true
    end
end

function initial_population!(opt, cache, bounds, f)
    opt_init = deepcopy(opt)
    opt_init.options.iterations = 2
    Metaheuristics.optimize(f, bounds, opt_init)

    pop_size = opt_init.parameters.N
    population_rand = [bounds[1, :] +
                       rand(length(cache.u0)) .* (bounds[2, :] - bounds[1, :])
                       for i in 1:(pop_size - 1)]
    push!(population_rand, cache.u0)
    population_init = [Metaheuristics.create_child(x, f(x)) for x in population_rand]
    prev_status = Metaheuristics.State(Metaheuristics.get_best(population_init),
        population_init)
    opt.parameters.N = pop_size
    opt.status = prev_status
    return nothing
end

function __map_optimizer_args!(cache::OptimizationBase.OptimizationCache,
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
        elseif j.first == :use_initial
            continue
        else
            error("$(j.first) keyword is not a valid option for $(typeof(opt).super) algorithm.")
        end
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

function SciMLBase.__init(prob::SciMLBase.OptimizationProblem,
        opt::Metaheuristics.AbstractAlgorithm; use_initial = false,
        callback = (args...) -> (false),
        progress = false, kwargs...)
    return OptimizationCache(prob, opt; use_initial = use_initial,
        callback = callback,
        progress = progress,
        kwargs...)
end

function SciMLBase.__solve(cache::OptimizationCache{O}) where {O <:
                                                    Metaheuristics.AbstractAlgorithm}
    local x

    maxiters = OptimizationBase._check_and_convert_maxiters(cache.solver_args.maxiters)
    maxtime = OptimizationBase._check_and_convert_maxtime(cache.solver_args.maxtime)

    f = cache.f
    _loss = function (θ)
        if isa(f, MultiObjectiveOptimizationFunction)
            return cache.f(θ, cache.p)
        else
            x = cache.f(θ, cache.p)
            return first(x)
        end
    end

    if !isnothing(cache.lb) & !isnothing(cache.ub)
        opt_bounds = [cache.lb cache.ub]'
    end

    if !isnothing(cache.f.cons)
        @warn "Equality constraints are current not passed on by Optimization"
    end

    if !isnothing(cache.lcons)
        @warn "Inequality constraints are current not passed on by Optimization"
    end

    if !isnothing(cache.ucons)
        @warn "Inequality constraints are current not passed on by Optimization"
    end

    __map_optimizer_args!(
        cache, cache.opt; callback = cache.callback, cache.solver_args...,
        maxiters = maxiters,
        maxtime = maxtime)

    if cache.solver_args.use_initial
        initial_population!(cache.opt, cache, opt_bounds, _loss)
    end

    t0 = time()
    opt_res = Metaheuristics.optimize(_loss, opt_bounds, cache.opt)
    t1 = time()
    stats = OptimizationBase.OptimizationStats(; time = t1 - t0)
    SciMLBase.build_solution(cache, cache.opt,
        Metaheuristics.minimizer(opt_res),
        Metaheuristics.minimum(opt_res); original = opt_res,
        stats = stats)
end

end
