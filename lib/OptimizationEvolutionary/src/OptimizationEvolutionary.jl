module OptimizationEvolutionary

using Reexport
@reexport using Evolutionary, Optimization
using Optimization.SciMLBase

SciMLBase.allowsbounds(opt::Evolutionary.AbstractOptimizer) = true
SciMLBase.allowsconstraints(opt::Evolutionary.AbstractOptimizer) = true
SciMLBase.supports_opt_cache_interface(opt::Evolutionary.AbstractOptimizer) = true
SciMLBase.requiresgradient(opt::Evolutionary.AbstractOptimizer) = false
SciMLBase.requiresgradient(opt::Evolutionary.NSGA2) = false
SciMLBase.requireshessian(opt::Evolutionary.AbstractOptimizer) = false
SciMLBase.requiresconsjac(opt::Evolutionary.AbstractOptimizer) = false
SciMLBase.requiresconshess(opt::Evolutionary.AbstractOptimizer) = false

decompose_trace(trace::Evolutionary.OptimizationTrace) = last(trace)
decompose_trace(trace::Evolutionary.OptimizationTraceRecord) = trace

# Overload the trace! function to add the population to the trace prior to calling any user-defined trace! method
function Evolutionary.trace!(tr, iteration, objfun, state, population,
        method::Evolutionary.AbstractOptimizer, options, curr_time = time())
    dt = Dict{String, Any}()
    dt["time"] = curr_time

    # record current u0. Needed for constructing OptimizationState.
    dt["curr_u"] = population[end]

    # set additional trace value
    Evolutionary.trace!(dt, objfun, state, population, method, options)
    Evolutionary.update!(tr,
        state,
        iteration,
        Evolutionary.value(state),
        dt,
        options.store_trace,
        options.show_trace,
        options.show_every,
        options.callback)
end

function __map_optimizer_args(cache::OptimizationCache,
        opt::Evolutionary.AbstractOptimizer;
        callback = nothing,
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        kwargs...)
    mapped_args = (; kwargs...)

    if !isnothing(callback)
        mapped_args = (; mapped_args..., callback = callback)
    end

    if !isnothing(maxiters)
        mapped_args = (; mapped_args..., iterations = maxiters)
    end

    if !isnothing(maxtime)
        mapped_args = (; mapped_args..., time_limit = Float64(maxtime))
    end

    if !isnothing(abstol)
        mapped_args = (; mapped_args..., abstol = Float64(abstol))
    end

    if !isnothing(reltol)
        mapped_args = (; mapped_args..., reltol = Float64(reltol))
    end

    return Evolutionary.Options(; mapped_args...)
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
        C
}) where {
        F,
        RC,
        LB,
        UB,
        LC,
        UC,
        S,
        O <:
        Evolutionary.AbstractOptimizer,
        D,
        P,
        C
}
    local x, cur, state

    function _cb(trace)
        curr_u = decompose_trace(trace).metadata["curr_u"]
        opt_state = Optimization.OptimizationState(;
            iter = decompose_trace(trace).iteration,
            u = curr_u,
            objective = x[1],
            original = trace)
        cb_call = cache.callback(opt_state, decompose_trace(trace).value...)
        if !(cb_call isa Bool)
            error("The callback should return a boolean `halt` for whether to stop the optimization process.")
        end
        cb_call
    end

    maxiters = Optimization._check_and_convert_maxiters(cache.solver_args.maxiters)
    maxtime = Optimization._check_and_convert_maxtime(cache.solver_args.maxtime)

    f = cache.f

    _loss = function (θ)
        if isa(f, MultiObjectiveOptimizationFunction)
            x = f(θ, cache.p)
            return x
        else
            x = f(θ, cache.p)
            return first(x)
        end
    end

    opt_args = __map_optimizer_args(cache, cache.opt; callback = _cb, cache.solver_args...,
        maxiters = maxiters,
        maxtime = maxtime)

    t0 = time()
    if isnothing(cache.lb) || isnothing(cache.ub)
        if !isnothing(f.cons)
            c = x -> (res = zeros(length(cache.lcons)); f.cons(res, x); res)
            cons = WorstFitnessConstraints(Float64[], Float64[], cache.lcons, cache.ucons,
                c)
            if isa(f, MultiObjectiveOptimizationFunction)
                opt_res = Evolutionary.optimize(
                    _loss, _loss(cache.u0), cons, cache.u0, cache.opt, opt_args)
            else
                opt_res = Evolutionary.optimize(_loss, cons, cache.u0, cache.opt, opt_args)
            end
        else
            if isa(f, MultiObjectiveOptimizationFunction)
                opt_res = Evolutionary.optimize(
                    _loss, _loss(cache.u0), cache.u0, cache.opt, opt_args)
            else
                opt_res = Evolutionary.optimize(_loss, cache.u0, cache.opt, opt_args)
            end
        end
    else
        if !isnothing(f.cons)
            c = x -> (res = zeros(length(cache.lcons)); f.cons(res, x); res)
            cons = WorstFitnessConstraints(cache.lb, cache.ub, cache.lcons, cache.ucons, c)
        else
            cons = BoxConstraints(cache.lb, cache.ub)
        end
        if isa(f, MultiObjectiveOptimizationFunction)
            opt_res = Evolutionary.optimize(
                _loss, _loss(cache.u0), cons, cache.u0, cache.opt, opt_args)
        else
            opt_res = Evolutionary.optimize(_loss, cons, cache.u0, cache.opt, opt_args)
        end
    end
    t1 = time()
    opt_ret = Symbol(Evolutionary.converged(opt_res))
    stats = Optimization.OptimizationStats(; iterations = opt_res.iterations,
        time = t1 - t0, fevals = opt_res.f_calls)
    if !isa(f, MultiObjectiveOptimizationFunction)
        SciMLBase.build_solution(cache, cache.opt,
            Evolutionary.minimizer(opt_res),
            Evolutionary.minimum(opt_res); original = opt_res,
            retcode = opt_ret,
            stats = stats)
    else
        ans = Evolutionary.minimizer(opt_res)
        SciMLBase.build_solution(cache, cache.opt,
            ans,
            _loss(ans[1]); original = opt_res,
            retcode = opt_ret,
            stats = stats)
    end
end

end
