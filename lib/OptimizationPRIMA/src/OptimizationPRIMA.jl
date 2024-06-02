module OptimizationPRIMA

using Optimization, Optimization.SciMLBase, Reexport
@reexport using PRIMA

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
SciMLBase.requiresgradient(opt::Union{BOBYQA, LINCOA, COBYLA}) = true
SciMLBase.requiresconsjac(opt::Union{LINCOA, COBYLA}) = true

function Optimization.OptimizationCache(prob::SciMLBase.OptimizationProblem,
        opt::PRIMASolvers, data;
        callback = Optimization.DEFAULT_CALLBACK,
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        progress = false,
        kwargs...)
    reinit_cache = Optimization.ReInitCache(prob.u0, prob.p)
    num_cons = prob.ucons === nothing ? 0 : length(prob.ucons)
    if prob.f.adtype isa SciMLBase.NoAD && opt isa COBYLA
        throw("We evaluate the jacobian and hessian of the constraints once to automatically detect 
        linear and nonlinear constraints, please provide a valid AD backend for using COBYLA.")
    else
        f = Optimization.instantiate_function(
            prob.f, reinit_cache.u0, prob.f.adtype, reinit_cache.p, num_cons)
    end

    return Optimization.OptimizationCache(f, reinit_cache, prob.lb, prob.ub, prob.lcons,
        prob.ucons, prob.sense,
        opt, data, progress, callback,
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

function __map_optimizer_args!(cache::Optimization.OptimizationCache, opt::PRIMASolvers;
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

function SciMLBase.__solve(cache::Optimization.OptimizationCache{
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
        C
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
        C
}
    iter = 0
    _loss = function (θ)
        x = cache.f(θ, cache.p)
        iter += 1
        opt_state = Optimization.OptimizationState(u = θ, objective = x[1], iter = iter)
        if cache.callback(opt_state, x...)
            error("Optimization halted by callback.")
        end
        return x[1]
    end

    optfunc = get_solve_func(cache.opt)

    maxiters = Optimization._check_and_convert_maxiters(cache.solver_args.maxiters)
    maxtime = Optimization._check_and_convert_maxtime(cache.solver_args.maxtime)

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
        function fwcons(θ, res)
            nonlincons(res, θ)
            return _loss(θ)
        end
        (minx, minf, nf, rc, cstrv) = optfunc(fwcons,
            cache.u0;
            linear_eq = (A₁, b₁),
            linear_ineq = (A₂, b₂),
            nonlinear_ineq = length(nonlininds),
            kws...)
    elseif cache.opt isa LINCOA
        (minx, minf, nf, rc, cstrv) = optfunc(_loss, cache.u0; kws...)
    else
        (minx, minf, nf, rc) = optfunc(_loss, cache.u0; kws...)
    end
    t1 = time()

    retcode = sciml_prima_retcode(PRIMA.reason(rc))
    stats = Optimization.OptimizationStats(; time = t1 - t0, fevals = nf)
    SciMLBase.build_solution(cache, cache.opt, minx,
        minf; retcode = retcode,
        stats = stats)
end

export UOBYQA, NEWUOA, BOBYQA, LINCOA, COBYLA
end
