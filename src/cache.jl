
function Base.getproperty(cache::SciMLBase.AbstractOptimizationCache, x::Symbol)
    if x in fieldnames(Optimization.ReInitCache)
        return getfield(cache.reinit_cache, x)
    end
    return getfield(cache, x)
end

SciMLBase.has_reinit(cache::SciMLBase.AbstractOptimizationCache) = true
function SciMLBase.reinit!(cache::SciMLBase.AbstractOptimizationCache; p = missing,
                           u0 = missing)
    if p === missing && u0 === missing
        p, u0 = cache.p, cache.u0
    else # at least one of them has a value
        if p === missing
            p = cache.p
        end
        if u0 === missing
            u0 = cache.u0
        end
        if (eltype(p) <: Pair && !isempty(p)) || (eltype(u0) <: Pair && !isempty(u0)) # one is a non-empty symbolic map
            hasproperty(cache.f, :sys) && hasfield(typeof(cache.f.sys), :ps) ||
                throw(ArgumentError("This cache does not support symbolic maps with `remake`, i.e. it does not have a symbolic origin." *
                                    " Please use `remake` with the `p` keyword argument as a vector of values, paying attention to parameter order."))
            hasproperty(cache.f, :sys) && hasfield(typeof(cache.f.sys), :states) ||
                throw(ArgumentError("This cache does not support symbolic maps with `remake`, i.e. it does not have a symbolic origin." *
                                    " Please use `remake` with the `u0` keyword argument as a vector of values, paying attention to state order."))
            p, u0 = SciMLBase.process_p_u0_symbolic(cache, p, u0)
        end
    end

    cache.reinit_cache.p = p
    cache.reinit_cache.u0 = u0

    return cache
end

struct OptimizationCache{F, RC, LB, UB, LC, UC, S, O, D, P, C} <:
       SciMLBase.AbstractOptimizationCache
    f::F
    reinit_cache::RC
    lb::LB
    ub::UB
    lcons::LC
    ucons::UC
    sense::S
    opt::O
    data::D
    progress::P
    callback::C
    solver_args::NamedTuple
end

function OptimizationCache(prob::OptimizationProblem, opt, data; progress, callback,
                                  kwargs...)
    reinit_cache = Optimization.ReInitCache(prob.u0, prob.p)
    num_cons = prob.ucons === nothing ? 0 : length(prob.ucons)
    f = Optimization.instantiate_function(prob.f, reinit_cache, prob.f.adtype, num_cons)
    return OptimizationCache(f, reinit_cache, prob.lb, prob.ub, prob.lcons,
                                prob.ucons, prob.sense,
                                opt, data, progress, callback, NamedTuple(kwargs))
end

function SciMLBase.__init(prob::SciMLBase.OptimizationProblem, opt,
                          data = Optimization.DEFAULT_DATA;
                          callback = (args...) -> (false),
                          maxiters::Union{Number, Nothing} = nothing,
                          maxtime::Union{Number, Nothing} = nothing,
                          abstol::Union{Number, Nothing} = nothing,
                          reltol::Union{Number, Nothing} = nothing,
                          progress = false,
                          kwargs...)
    return OptimizationCache(prob, opt, data; maxiters, maxtime, abstol, callback,
                                reltol, progress,
                                kwargs...)
end
