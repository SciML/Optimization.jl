module OptimizationPRIMA

using PRIMA, Optimization, Optimization.SciMLBase

abstract type PRIMASolvers end

struct UOBYQA <: PRIMASolvers end
struct NEWUOA <: PRIMASolvers end
struct BOBYQA <: PRIMASolvers end
struct LINCOA <: PRIMASolvers end
struct COBYLA <: PRIMASolvers end

SciMLBase.supports_opt_cache_interface(::PRIMASolvers) = true
SciMLBase.allowsconstraints(::Union{LINCOA, COBYLA}) = true
SciMLBase.allowsbounds(opt::Union{BOBYQA, LINCOA, COBYLA}) = true
SciMLBase.requiresconstraints(opt::COBYLA) = true

function SciMLBase.__init(prob::SciMLBase.OptimizationProblem, opt::PRIMASolvers,
    data = Optimization.DEFAULT_DATA;
    callback = (args...) -> (false),
    progress = false, kwargs...)
    return OptimizationCache(prob, opt, data; callback, progress,
        kwargs...)
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

function __map_optimizer_args!(cache::OptimizationCache, opt::PRIMASolvers;
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
    if rc in ["SMALL_TR_RADIUS", "TRSUBP_FAILED","NAN_INF_X"
        ,"NAN_INF_F"
        ,"NAN_INF_MODEL"
        ,"DAMAGING_ROUNDING"
        ,"ZERO_LINEAR_CONSTRAINT"
        ,"INVALID_INPUT"
        ,"ASSERTION_FAILS"
        ,"VALIDATION_FAILS"
        ,"MEMORY_ALLOCATION_FAILS"]
        return ReturnCode.Failure
    else rc in [
        "FTARGET_ACHIEVED"
        "MAXFUN_REACHED"
        "MAXTR_REACHED"
        "NO_SPACE_BETWEEN_BOUNDS"
    ]
        return ReturnCode.Success
    end
end

function SciMLBase.__solve(cache::OptimizationCache{
    F,
    RC,
    LB,
    UB,
    LC,
    UC,
    S,
    O,
    D,
    P,
    C,
}) where {
    F,
    RC,
    LB,
    UB,
    LC,
    UC,
    S,
    O <: PRIMASolvers,
    D,
    P,
    C,
}
    _loss = function (θ)
        x = cache.f(θ, cache.p)
        if cache.callback(θ, x...)
            error("Optimization halted by callback.")
        end
        return x[1]
    end

   optfunc = get_solve_func(cache.opt)


    maxiters = Optimization._check_and_convert_maxiters(cache.solver_args.maxiters)
    maxtime = Optimization._check_and_convert_maxtime(cache.solver_args.maxtime)

    kws = __map_optimizer_args!(cache, cache.opt; callback = cache.callback, maxiters = maxiters,
        maxtime = maxtime,
        cache.solver_args...)

    t0 = time()
    if cache.opt isa COBYLA
        function fwcons(θ, res)
            cache.f.cons(res, θ, cache.p)
            return _loss(θ)
        end
        (minx, minf, nf, rc, cstrv) = optfunc(fwcons, cache.u0; kws...)
    elseif cache.opt isa LINCOA
        (minx, minf, nf, rc, cstrv) = optfunc(_loss, cache.u0; kws...)
    else
        (minx, minf, nf, rc) = optfunc(_loss, cache.u0; kws...)
    end
    t1 = time()

    retcode = sciml_prima_retcode(PRIMA.reason(rc))

    SciMLBase.build_solution(cache, cache.opt, minx,
        minf; retcode = retcode,
        solve_time = t1 - t0)
end

export UOBYQA, NEWUOA, BOBYQA, LINCOA, COBYLA
end
