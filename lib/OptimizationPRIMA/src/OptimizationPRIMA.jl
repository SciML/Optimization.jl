module OptimizationPRIMA

using OptimizationBase, SciMLBase, Reexport
@reexport using PRIMA

abstract type PRIMASolvers end

struct UOBYQA <: PRIMASolvers end
struct NEWUOA <: PRIMASolvers end
struct BOBYQA <: PRIMASolvers end
struct LINCOA <: PRIMASolvers end
struct COBYLA <: PRIMASolvers end

@static if isdefined(SciMLBase, :supports_opt_cache_interface)
    SciMLBase.supports_opt_cache_interface(::PRIMASolvers) = true
end
@static if isdefined(OptimizationBase, :supports_opt_cache_interface)
    OptimizationBase.supports_opt_cache_interface(::PRIMASolvers) = true
end
SciMLBase.allowsconstraints(::Union{LINCOA, COBYLA}) = true
SciMLBase.allowsbounds(opt::Union{BOBYQA, LINCOA, COBYLA}) = true
SciMLBase.requiresconstraints(opt::COBYLA) = true
SciMLBase.requiresconsjac(opt::COBYLA) = true
SciMLBase.requiresconshess(opt::COBYLA) = true

function OptimizationBase.OptimizationCache(prob::SciMLBase.OptimizationProblem,
        opt::PRIMASolvers;
        callback = OptimizationBase.DEFAULT_CALLBACK,
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        progress = false,
        kwargs...)
    reinit_cache = OptimizationBase.ReInitCache(prob.u0, prob.p)
    num_cons = prob.ucons === nothing ? 0 : length(prob.ucons)
    if prob.f.adtype isa SciMLBase.NoAD && opt isa COBYLA
        throw("We evaluate the jacobian and hessian of the constraints once to automatically detect
        linear and nonlinear constraints, please provide a valid AD backend for using COBYLA.")
    else
        if opt isa COBYLA
            f = OptimizationBase.instantiate_function(
                prob.f, reinit_cache.u0, prob.f.adtype, reinit_cache.p, num_cons,
                cons_j = true, cons_h = true)
        else
            f = OptimizationBase.instantiate_function(
                prob.f, reinit_cache.u0, prob.f.adtype, reinit_cache.p, num_cons)
        end
    end

    return OptimizationBase.OptimizationCache(
        f, reinit_cache, prob.lb, prob.ub, prob.lcons,
        prob.ucons, prob.sense,
        opt, progress, callback, nothing,
        OptimizationBase.OptimizationBase.AnalysisResults(nothing, nothing),
        merge((; maxiters, maxtime, abstol, reltol),
            NamedTuple(kwargs)))
end

function get_solve_func(opt::PRIMASolvers)
    if opt isa UOBYQA
        return PRIMA.uobyqa
    elseif opt isa NEWUOA
        return PRIMA.newuoa
    elseif opt isa BOBYQA
        return PRIMA.bobyqa
    elseif opt isa LINCOA
        return PRIMA.lincoa
    elseif opt isa COBYLA
        return PRIMA.cobyla
    end
end

function __map_optimizer_args!(
        cache::OptimizationBase.OptimizationCache, opt::PRIMASolvers;
        callback = nothing,
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        kwargs...)
    kws = (; kwargs...)

    if !isnothing(maxiters)
        kws = (; kws..., maxfun = maxiters)
    end

    if cache.ub !== nothing
        kws = (; kws..., xu = cache.ub, xl = cache.lb)
    end

    if !isnothing(maxtime) || !isnothing(abstol) || !isnothing(reltol)
        error("maxtime, abstol and reltol kwargs not supported in $opt")
    end

    return kws
end

function sciml_prima_retcode(rc::AbstractString)
    if rc in [
        "SMALL_TR_RADIUS", "TRSUBP_FAILED", "NAN_INF_X", "NAN_INF_F", "NAN_INF_MODEL",
        "DAMAGING_ROUNDING", "ZERO_LINEAR_CONSTRAINT", "INVALID_INPUT", "ASSERTION_FAILS",
        "VALIDATION_FAILS", "MEMORY_ALLOCATION_FAILS"]
        return ReturnCode.Failure
    else
        rc in ["FTARGET_ACHIEVED"
               "MAXFUN_REACHED"
               "MAXTR_REACHED"
               "NO_SPACE_BETWEEN_BOUNDS"]
        return ReturnCode.Success
    end
end

function SciMLBase.__solve(cache::OptimizationCache{O}) where {O <: PRIMASolvers}
    iter = 0
    _loss = function (θ)
        x = cache.f(θ, cache.p)
        iter += 1
        opt_state = OptimizationBase.OptimizationState(
            u = θ, p = cache.p, objective = x[1], iter = iter)
        if cache.callback(opt_state, x...)
            error("Optimization halted by callback.")
        end
        return x[1]
    end

    optfunc = get_solve_func(cache.opt)

    maxiters = OptimizationBase._check_and_convert_maxiters(cache.solver_args.maxiters)
    maxtime = OptimizationBase._check_and_convert_maxtime(cache.solver_args.maxtime)

    kws = __map_optimizer_args!(cache, cache.opt; callback = cache.callback,
        maxiters = maxiters,
        maxtime = maxtime,
        cache.solver_args...)

    t0 = time()
    if cache.opt isa COBYLA
        lineqsinds = Int[]
        linineqsinds = Int[]
        nonlininds = Int[]
        H = [zeros(length(cache.u0), length(cache.u0)) for i in 1:length(cache.lcons)]
        J = zeros(length(cache.lcons), length(cache.u0))

        cache.f.cons_h(H, ones(length(cache.u0)))
        cache.f.cons_j(J, ones(length(cache.u0)))
        for i in eachindex(cache.lcons)
            if iszero(H[i]) && cache.lcons[i] == cache.ucons[i]
                push!(lineqsinds, i)
            elseif iszero(H[i]) && cache.lcons[i] != cache.ucons[i]
                push!(linineqsinds, i)
            else
                push!(nonlininds, i)
            end
        end
        res1 = zeros(length(cache.lcons))
        nonlincons = (res, θ) -> (cache.f.cons(res1, θ); res .= res1[nonlininds])
        A₁ = J[lineqsinds, :]
        b₁ = cache.lcons[lineqsinds]
        A₂ = J[linineqsinds, :]
        b₂ = cache.ucons[linineqsinds]

        (minx,
        inf) = optfunc(_loss,
            cache.u0;
            linear_eq = (A₁, b₁),
            linear_ineq = (A₂, b₂),
            nonlinear_ineq = x -> (res = zeros(eltype(x), length(nonlininds));
            nonlincons(
                res, x);
            res),
            kws...)
    else
        (minx, inf) = optfunc(_loss, cache.u0; kws...)
    end
    t1 = time()

    retcode = sciml_prima_retcode(PRIMA.reason(inf))
    stats = OptimizationBase.OptimizationStats(; time = t1 - t0, fevals = inf.nf)
    SciMLBase.build_solution(cache, cache.opt, minx,
        inf.fx; retcode = retcode,
        stats = stats, original = inf)
end

export UOBYQA, NEWUOA, BOBYQA, LINCOA, COBYLA
end
