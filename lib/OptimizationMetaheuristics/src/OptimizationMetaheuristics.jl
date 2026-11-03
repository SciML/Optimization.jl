module OptimizationMetaheuristics

using Metaheuristics
# The Metaheuristics algorithm names are this package's whole point: `using
# Optimization, OptimizationMetaheuristics` has to be enough to write
# `solve(prob, ECA())`, exactly as the docs list them. They are re-surfaced by name
# below rather than by blanket `@reexport`, which also exported `optimize!`/`minimizer`
# /`Options` and the MCDM decision-making surface. Everything stays owned and
# documented upstream in Metaheuristics.jl.
using Metaheuristics: ABC, BRKGA, CCMO, CGSA, CSO, DE, ECA, GA, GRASP, MCCGA,
    MOEAD_DE, NSGA2, NSGA3, PSO, RDEx, SA, SHADE, SMS_EMOA, SPEA2, VND, VNS, WOA, εDE
# Not re-exported: the optimization API comes from `Optimization`/`OptimizationBase`,
# which the user loads directly. This package's public surface is its own solvers.
using OptimizationBase
using SciMLLogging: @SciMLMessage
using SciMLBase

export Metaheuristics

# Metaheuristics' algorithms; approved via `reexports_allow` in test/qa/qa.jl. The
# configuration objects stay qualified as the docs spell them (`Metaheuristics.Options`,
# `Metaheuristics.Information`) — `Options` is far too generic a name to export.
export ABC, BRKGA, CCMO, CGSA, CSO, DE, ECA, GA, GRASP, MCCGA,
    MOEAD_DE, NSGA2, NSGA3, PSO, RDEx, SA, SHADE, SMS_EMOA, SPEA2, VND, VNS, WOA, εDE

SciMLBase.requiresbounds(opt::Metaheuristics.AbstractAlgorithm) = true
SciMLBase.allowsbounds(opt::Metaheuristics.AbstractAlgorithm) = true
SciMLBase.allowscallback(opt::Metaheuristics.AbstractAlgorithm) = false
# Algorithms marked as supporting constraints in the algorithm table of the Metaheuristics.jl
# documentation. The others ignore the constraint violations returned by the objective.
# `BRKGA()` constructs a `GA`.
const ConstrainedAlgorithm = Metaheuristics.Algorithm{
    <:Union{
        CCMO, CSO, DE, ECA, GA, NSGA2, NSGA3, PSO, SA, SHADE, SMS_EMOA, SPEA2, εDE,
    },
}
SciMLBase.allowsconstraints(opt::ConstrainedAlgorithm) = true
SciMLBase.requiresconsjac(opt::Metaheuristics.AbstractAlgorithm) = false
SciMLBase.requiresconshess(opt::Metaheuristics.AbstractAlgorithm) = false
SciMLBase.has_init(opt::Metaheuristics.AbstractAlgorithm) = true

function initial_population!(opt, cache, bounds, f)
    opt_init = deepcopy(opt)
    opt_init.options.iterations = 2
    Metaheuristics.optimize(f, bounds, opt_init)

    pop_size = opt_init.parameters.N
    population_rand = [
        bounds[1, :] +
            rand(length(cache.u0)) .* (bounds[2, :] - bounds[1, :])
            for i in 1:(pop_size - 1)
    ]
    push!(population_rand, cache.u0)
    population_init = [Metaheuristics.create_child(x, f(x)) for x in population_rand]
    prev_status = Metaheuristics.State(
        Metaheuristics.get_best(population_init),
        population_init
    )
    opt.parameters.N = pop_size
    opt.status = prev_status
    return nothing
end

# Return a function mapping `θ` to the constraint values `(g, h)` in the form expected by
# Metaheuristics.jl, i.e., `g(θ) ≤ 0` and `h(θ) = 0`, from `lcons ≤ cons(θ) ≤ ucons`. A
# constraint with `lcons[i] == ucons[i]` is an equality constraint, and every finite bound of
# the remaining constraints is an inequality constraint.
function constraint_violations(cache)
    lcons, ucons = cache.lcons, cache.ucons
    eq = findall(lcons .== ucons)
    upper = findall(i -> lcons[i] != ucons[i] && isfinite(ucons[i]), eachindex(ucons))
    lower = findall(i -> lcons[i] != ucons[i] && isfinite(lcons[i]), eachindex(lcons))
    res = zeros(eltype(cache.u0), length(lcons))
    return function (θ)
        cache.f.cons(res, θ)
        g = [res[upper] .- ucons[upper]; lcons[lower] .- res[lower]]
        h = res[eq] .- ucons[eq]
        return g, h
    end
end

function __map_optimizer_args!(
        cache::OptimizationBase.OptimizationCache,
        opt::Metaheuristics.AbstractAlgorithm;
        callback = nothing,
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        kwargs...
    )
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
        @SciMLMessage(
            lazy"common reltol is currently not used by $(typeof(opt).super)",
            cache.verbose, :unsupported_kwargs
        )
    end
    return nothing
end

function SciMLBase.__init(
        prob::SciMLBase.OptimizationProblem,
        opt::Metaheuristics.AbstractAlgorithm; use_initial = false,
        callback = OptimizationBase.DEFAULT_CALLBACK,
        progress = false, kwargs...
    )
    return OptimizationCache(
        prob, opt; use_initial = use_initial,
        callback = callback,
        progress = progress,
        kwargs...
    )
end

function SciMLBase.__solve(cache::OptimizationCache{O}) where {
        O <:
        Metaheuristics.AbstractAlgorithm,
    }
    local x

    maxiters = OptimizationBase._check_and_convert_maxiters(cache.solver_args.maxiters)
    maxtime = OptimizationBase._check_and_convert_maxtime(cache.solver_args.maxtime)

    f = cache.f
    violations = isnothing(f.cons) ? nothing : constraint_violations(cache)
    _loss = function (θ)
        if isa(f, MultiObjectiveOptimizationFunction)
            fx = cache.f(θ, cache.p)
            isnothing(violations) && return fx
            # The objective returns the objective values and its own constraint values
            # `(F, g, h)`, to which the constraints of the problem are appended.
            F, g, h = fx
            gc, hc = violations(θ)
            return F, [g; gc], [h; hc]
        else
            fx = first(cache.f(θ, cache.p))
            isnothing(violations) && return fx
            g, h = violations(θ)
            return fx, g, h
        end
    end

    opt_bounds = nothing
    if !isnothing(cache.lb) & !isnothing(cache.ub)
        opt_bounds = [cache.lb cache.ub]'
    end

    __map_optimizer_args!(
        cache, cache.opt; callback = cache.callback, cache.solver_args...,
        maxiters = maxiters,
        maxtime = maxtime
    )

    if cache.solver_args.use_initial
        initial_population!(cache.opt, cache, opt_bounds, _loss)
    end

    t0 = time()
    opt_res = Metaheuristics.optimize(_loss, opt_bounds, cache.opt)
    t1 = time()
    stats = OptimizationBase.OptimizationStats(; time = t1 - t0)
    return SciMLBase.build_solution(
        cache, cache.opt,
        Metaheuristics.minimizer(opt_res),
        Metaheuristics.minimum(opt_res); original = opt_res,
        stats = stats
    )
end

end
