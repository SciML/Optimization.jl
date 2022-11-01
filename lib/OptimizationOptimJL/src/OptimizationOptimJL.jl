module OptimizationOptimJL

using Reexport
@reexport using Optim, Optimization
using Optimization.SciMLBase, SparseArrays
decompose_trace(trace::Optim.OptimizationTrace) = last(trace)
decompose_trace(trace::Optim.OptimizationState) = trace

SciMLBase.allowsconstraints(::IPNewton) = true
SciMLBase.requiresconstraints(::IPNewton) = true
SciMLBase.allowsbounds(opt::Optim.AbstractOptimizer) = true
SciMLBase.allowsbounds(opt::Optim.SimulatedAnnealing) = false
SciMLBase.requiresbounds(opt::Optim.Fminbox) = true
SciMLBase.requiresbounds(opt::Optim.SAMIN) = true

abstract type AbstractOptimJLOptimizationCache <: SciMLBase.AbstractOptimizationCache end

struct OptimJLOptimizationCache{F, RC, LB, UB, S, O, D} <: AbstractOptimJLOptimizationCache
    f::F
    reinit_cache::RC
    lb::LB
    ub::UB
    sense::S
    opt::O
    data::D
    solver_args::NamedTuple
end

struct OptimJLBoxConstraintOptimizationCache{F, RC, LB, UB, S, O, D} <: AbstractOptimJLOptimizationCache
    f::F
    reinit_cache::RC
    lb::LB
    ub::UB
    sense::S
    opt::O
    data::D
    solver_args::NamedTuple
end

struct OptimJLConstraintOptimizationCache{F, FC, RC, LB, UB, S, O, D} <: AbstractOptimJLOptimizationCache
    f::F
    fc::FC
    reinit_cache::RC
    lb::LB
    ub::UB
    sense::S
    opt::O
    data::D
    solver_args::NamedTuple
end

function Base.getproperty(cache::AbstractOptimJLOptimizationCache, x::Symbol)
    if x in fieldnames(Optimization.ReInitCache)
        return getfield(cache.reinit_cache, x)
    end
    return getfield(cache, x)
end

function OptimJLOptimizationCache(prob::OptimizationProblem, opt, data; kwargs...)
    reinit_cache = Optimization.ReInitCache(prob.u0, prob.p) # everything that can be changed via `reinit`
    f = Optimization.instantiate_function(prob.f, reinit_cache, prob.f.adtype)

    !(opt isa Optim.ZerothOrderOptimizer) && f.grad === nothing &&
        error("Use OptimizationFunction to pass the derivatives or automatically generate them with one of the autodiff backends")

    _loss = function (θ)
        x = f.f(θ, reinit_cache.p, cur...)
        __x = first(x)
        return prob.sense === Optimization.MaxSense ? -__x : __x
    end

    fg! = function (G, θ)
        if G !== nothing
            f.grad(G, θ, cur...)
            if prob.sense === Optimization.MaxSense
                G .*= false
            end
        end
        return _loss(θ)
    end

    if opt isa Optim.KrylovTrustRegion
        hv = function (H, θ, v)
            f.hv(H, θ, v, cur...)
            if prob.sense === Optimization.MaxSense
                H .*= false
            end
        end
        optim_f = Optim.TwiceDifferentiableHV(_loss, fg!, hv, reinit_cache.u0)
    else
        gg = function (G, θ)
            f.grad(G, θ, cur...)
            if prob.sense === Optimization.MaxSense
                G .*= false
            end
        end

        hh = function (H, θ)
            f.hess(H, θ, cur...)
            if prob.sense === Optimization.MaxSense
                H .*= false
            end
        end
        F = eltype(reinit_cache.u0)
        optim_f = Optim.TwiceDifferentiable(_loss, gg, fg!, hh, reinit_cache.u0, real(zero(F)),
                                            Optim.NLSolversBase.alloc_DF(reinit_cache.u0,
                                                                         real(zero(F))),
                                            isnothing(f.hess_prototype) ?
                                            Optim.NLSolversBase.alloc_H(reinit_cache.u0,
                                                                        real(zero(F))) :
                                            convert.(F, f.hess_prototype))
    end

    return OptimJLOptimizationCache(optim_f, reinit_cache, prob.lb, prob.ub, prob.sense, opt, data, NamedTuple(kwargs))
end

function OptimJLBoxConstraintOptimizationCache(prob::OptimizationProblem, opt, data; kwargs...)
    reinit_cache = Optimization.ReInitCache(prob.u0, prob.p) # everything that can be changed via `reinit`
    f = Optimization.instantiate_function(prob.f, reinit_cache, prob.f.adtype)

    !(opt isa Optim.ZerothOrderOptimizer) && f.grad === nothing &&
        error("Use OptimizationFunction to pass the derivatives or automatically generate them with one of the autodiff backends")

    _loss = function (θ)
        x = f.f(θ, reinit_cache.p, cur...)
        __x = first(x)
        return prob.sense === Optimization.MaxSense ? -__x : __x
    end
    fg! = function (G, θ)
        if G !== nothing
            f.grad(G, θ, cur...)
            if prob.sense === Optimization.MaxSense
                G .*= false
            end
        end
        return _loss(θ)
    end

    gg = function (G, θ)
        f.grad(G, θ, cur...)
        if prob.sense === Optimization.MaxSense
            G .*= false
        end
    end
    optim_f = Optim.OnceDifferentiable(_loss, gg, fg!, reinit_cache.u0)

    return OptimJLBoxConstraintOptimizationCache(optim_f, reinit_cache, prob.lb, prob.ub, prob.sense, opt, data, NamedTuple(kwargs))
end

function OptimJLConstraintOptimizationCache(prob::OptimizationProblem, opt, data; kwargs...)
    reinit_cache = Optimization.ReInitCache(prob.u0, prob.p) # everything that can be changed via `reinit`
    f = Optimization.instantiate_function(prob.f, reinit_cache, prob.f.adtype)

    f.cons_j === nothing &&
            error("This optimizer requires derivative definitions for nonlinear constraints. If the problem does not have nonlinear constraints, choose a different optimizer. Otherwise define the derivative for cons using OptimizationFunction either directly or automatically generate them with one of the autodiff backends")

    _loss = function (θ)
        x = f.f(θ, reinit_cache.p, cur...)
        __x = first(x)
        return prob.sense === Optimization.MaxSense ? -__x : __x
    end
    fg! = function (G, θ)
        if G !== nothing
            f.grad(G, θ, cur...)
            if prob.sense === Optimization.MaxSense
                G .*= false
            end
        end
        return _loss(θ)
    end
    gg = function (G, θ)
        f.grad(G, θ, cur...)
        if prob.sense === Optimization.MaxSense
            G .*= false
        end
    end

    hh = function (H, θ)
        f.hess(H, θ, cur...)
        if prob.sense === Optimization.MaxSense
            H .*= false
        end
    end
    F = eltype(reinit_cache.u0)
    optim_f = Optim.TwiceDifferentiable(_loss, gg, fg!, hh, reinit_cache.u0, real(zero(F)),
                                        Optim.NLSolversBase.alloc_DF(reinit_cache.u0,
                                                                     real(zero(F))),
                                        isnothing(f.hess_prototype) ?
                                        Optim.NLSolversBase.alloc_H(reinit_cache.u0,
                                                                    real(zero(F))) :
                                        convert.(F, f.hess_prototype))

    cons_hl! = function (h, θ, λ)
        res = [similar(h) for i in 1:length(λ)]
        f.cons_h(res, θ)
        for i in 1:length(λ)
            h .+= λ[i] * res[i]
        end
    end

    lb = prob.lb === nothing ? [] : prob.lb
    ub = prob.ub === nothing ? [] : prob.ub
    optim_fc = Optim.TwiceDifferentiableConstraints(f.cons, f.cons_j, cons_hl!, lb, ub,
                                                    prob.lcons, prob.ucons)

    return OptimJLConstraintOptimizationCache(optim_f, optim_fc, reinit_cache, prob.lb, prob.ub, prob.sense, opt, data, NamedTuple(kwargs))
end

SciMLBase.supports_opt_cache_interface(opt::Optim.AbstractOptimizer) = true
SciMLBase.supports_opt_cache_interface(opt::Union{Optim.Fminbox, Optim.SAMIN}) = true
SciMLBase.supports_opt_cache_interface(opt::Optim.ConstrainedOptimizer) = true

function __map_optimizer_args(cache::OptimJLOptimizationCache,
                              opt::Union{Optim.AbstractOptimizer, Optim.Fminbox,
                                         Optim.SAMIN, Optim.ConstrainedOptimizer};
                              callback = nothing,
                              maxiters::Union{Number, Nothing} = nothing,
                              maxtime::Union{Number, Nothing} = nothing,
                              abstol::Union{Number, Nothing} = nothing,
                              reltol::Union{Number, Nothing} = nothing,
                              kwargs...)
    if !isnothing(abstol)
        @warn "common abstol is currently not used by $(opt)"
    end

    mapped_args = (; extended_trace = true, kwargs...)

    if !isnothing(callback)
        mapped_args = (; mapped_args..., callback = callback)
    end

    if !isnothing(maxiters)
        mapped_args = (; mapped_args..., iterations = maxiters)
    end

    if !isnothing(maxtime)
        mapped_args = (; mapped_args..., time_limit = maxtime)
    end

    if !isnothing(reltol)
        mapped_args = (; mapped_args..., f_tol = reltol)
    end

    return Optim.Options(; mapped_args...)
end

function SciMLBase.__init(prob::OptimizationProblem, opt::Optim.AbstractOptimizer,
                  data = Optimization.DEFAULT_DATA;
                  callback = (args...) -> (false),
                  maxiters::Union{Number, Nothing} = nothing,
                  maxtime::Union{Number, Nothing} = nothing,
                  abstol::Union{Number, Nothing} = nothing,
                  reltol::Union{Number, Nothing} = nothing,
                  progress = false,
                  kwargs...)
    if !isnothing(prob.lb) || !isnothing(prob.ub)
        if !(opt isa Union{Optim.Fminbox, Optim.SAMIN, Optim.AbstractConstrainedOptimizer})
            if opt isa Optim.ParticleSwarm
                opt = Optim.ParticleSwarm(; lower = prob.lb, upper = prob.ub,
                                          n_particles = opt.n_particles)
            elseif opt isa Optim.SimulatedAnnealing
                @warn "$(opt) can currently not be wrapped in Fminbox(). The lower and upper bounds thus will be ignored. Consider using a different optimizer or open an issue with Optim.jl"
            else
                opt = Optim.Fminbox(opt)
            end
        end
    end
    return ___init(prob, opt, data; callback, maxiters, maxtime, abstol, reltol, progress, kwargs...)
end

function ___init(prob::OptimizationProblem, opt::Optim.AbstractOptimizer, args...; kwargs...)
end
function ___init(prob::OptimizationProblem, opt::Union{Optim.Fminbox, Optim.SAMIN}, args...; kwargs...)
end
function ___init(prob::OptimizationProblem, opt::Optim.ConstrainedOptimizer, args...; kwargs...)
end

function SciMLBase.__solve(cache::OptimJLOptimizationCache)
    local x, cur, state

    if cache.data != Optimization.DEFAULT_DATA
        maxiters = length(cache.data)
    else
        maxiters = cache.solver_args.maxiters
    end

    cur, state = iterate(cache.data)

    function _cb(trace)
        cb_call = opt isa Optim.NelderMead ?
                  cache.solver_args.callback(decompose_trace(trace).metadata["centroid"], x...) :
                  cache.solver_args.callback(decompose_trace(trace).metadata["x"], x...)
        if !(typeof(cb_call) <: Bool)
            error("The callback should return a boolean `halt` for whether to stop the optimization process.")
        end
        nx_itr = iterate(cache.data, state)
        if isnothing(nx_itr)
            true
        else
            cur, state = nx_itr
            cb_call
        end
    end

    maxiters = Optimization._check_and_convert_maxiters(maxiters)
    maxtime = Optimization._check_and_convert_maxtime(maxtime) 

    opt_args = __map_optimizer_args(cache, cache.opt, callback = _cb, maxiters = maxiters,
                                    maxtime = maxtime, abstol = cache.solver_args.abstol, reltol = cache.solver_args.reltol;
                                    cache.solver_args...)

    t0 = time()
    opt_res = Optim.optimize(cache.f, cache.u0, cache.opt, opt_args)
    t1 = time()
    opt_ret = Symbol(Optim.converged(opt_res))

    SciMLBase.build_solution(cache, cache.opt,
                             opt_res.minimizer,
                             cache.sense === Optimization.MaxSense ? -opt_res.minimum :
                             opt_res.minimum; original = opt_res, retcode = opt_ret,
                             solve_time = t1 - t0)
end

function SciMLBase.__solve(cache::OptimJLBoxConstraintOptimizationCache)
    local x, cur, state

    if cache.data != Optimization.DEFAULT_DATA
        maxiters = length(cache.data)
    else
        maxiters = cache.solver_args.maxiters
    end

    cur, state = iterate(cache.data)

    function _cb(trace)
        cb_call = !(opt isa Optim.SAMIN) && opt.method == Optim.NelderMead() ?
                  cache.solver_args.callback(decompose_trace(trace).metadata["centroid"], x...) :
                  cache.solver_args.callback(decompose_trace(trace).metadata["x"], x...)
        if !(typeof(cb_call) <: Bool)
            error("The callback should return a boolean `halt` for whether to stop the optimization process.")
        end
        nx_itr = iterate(cache.data, state)
        if isnothing(nx_itr)
            true
        else
            cur, state = nx_itr
            cb_call
        end
    end

    maxiters = Optimization._check_and_convert_maxiters(maxiters)
    maxtime = Optimization._check_and_convert_maxtime(cache.solver_args.maxtime)
    opt_args = __map_optimizer_args(cache, opt, callback = _cb, maxiters = maxiters,
                                    maxtime = maxtime, abstol = cache.solver_args.abstol, reltol = cache.solver_args.reltol;
                                    cache.solver_args...)

    t0 = time()
    opt_res = Optim.optimize(cache.f, cache.lb, cache.ub, cache.u0, cache.opt, opt_args)
    t1 = time()
    opt_ret = Symbol(Optim.converged(opt_res))

    SciMLBase.build_solution(cache, cache.opt,
                             opt_res.minimizer, opt_res.minimum;
                             original = opt_res, retcode = opt_ret, solve_time = t1 - t0)
end

function SciMLBase.__solve(cache::OptimJLConstraintOptimizationCache)
    local x, cur, state

    if cache.data != Optimization.DEFAULT_DATA
        maxiters = length(cache.data)
    else
        maxiters = cache.solver_args.maxiters
    end

    cur, state = iterate(cache.data)

    function _cb(trace)
        cb_call = cache.solver_args.callback(decompose_trace(trace).metadata["x"], x...)
        if !(typeof(cb_call) <: Bool)
            error("The callback should return a boolean `halt` for whether to stop the optimization process.")
        end
        nx_itr = iterate(cache.data, state)
        if isnothing(nx_itr)
            true
        else
            cur, state = nx_itr
            cb_call
        end
    end

    maxiters = Optimization._check_and_convert_maxiters(maxiters)
    maxtime = Optimization._check_and_convert_maxtime(cache.solver_args.maxtime)
    opt_args = __map_optimizer_args(cache, opt, callback = _cb, maxiters = maxiters,
                                    maxtime = maxtime, abstol = cache.solver_args.abstol, reltol = cache.solver_args.reltol;
                                    cache.solver_args...)

    t0 = time()
    opt_res = Optim.optimize(cache.f, cache.fc, cache.u0, opt, opt_args)
    t1 = time()
    opt_ret = Symbol(Optim.converged(opt_res))

    SciMLBase.build_solution(cache, opt,
                             opt_res.minimizer, opt_res.minimum;
                             original = opt_res, retcode = opt_ret)
end

end
