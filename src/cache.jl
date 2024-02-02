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

function OptimizationCache(prob::SciMLBase.OptimizationProblem, opt, data;
        callback = Optimization.DEFAULT_CALLBACK,
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        progress = false,
        kwargs...)
    reinit_cache = Optimization.ReInitCache(prob.u0, prob.p)
    num_cons = prob.ucons === nothing ? 0 : length(prob.ucons)
    f = Optimization.instantiate_function(prob.f, reinit_cache, prob.f.adtype, num_cons)
    return OptimizationCache(f, reinit_cache, prob.lb, prob.ub, prob.lcons,
        prob.ucons, prob.sense,
        opt, data, progress, callback,
        merge((; maxiters, maxtime, abstol, reltol),
            NamedTuple(kwargs)))
end

function SciMLBase.__init(prob::SciMLBase.OptimizationProblem, opt,
        data = Optimization.DEFAULT_DATA;
        callback = Optimization.DEFAULT_CALLBACK,
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
