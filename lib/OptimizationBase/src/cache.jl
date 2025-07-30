isa_dataiterator(data) = false

struct AnalysisResults{O, C}
    objective::O
    constraints::C
end

struct OptimizationCache{F, RC, LB, UB, LC, UC, S, O, P, C, M} <:
       SciMLBase.AbstractOptimizationCache
    f::F
    reinit_cache::RC
    lb::LB
    ub::UB
    lcons::LC
    ucons::UC
    sense::S
    opt::O
    progress::P
    callback::C
    manifold::M
    analysis_results::AnalysisResults
    solver_args::NamedTuple
end

function OptimizationCache(prob::SciMLBase.OptimizationProblem, opt;
        callback = DEFAULT_CALLBACK,
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        progress = false,
        structural_analysis = false,
        manifold = nothing,
        kwargs...)
    if isa_dataiterator(prob.p)
        reinit_cache = OptimizationBase.ReInitCache(prob.u0, iterate(prob.p)[1])
        reinit_cache_passedon = OptimizationBase.ReInitCache(prob.u0, prob.p)
    else
        reinit_cache = OptimizationBase.ReInitCache(prob.u0, prob.p)
        reinit_cache_passedon = reinit_cache
    end

    num_cons = prob.ucons === nothing ? 0 : length(prob.ucons)

    if !(prob.f.adtype isa DifferentiationInterface.SecondOrder ||
         prob.f.adtype isa AutoZygote) &&
       (SciMLBase.requireshessian(opt) || SciMLBase.requiresconshess(opt) ||
        SciMLBase.requireslagh(opt))
        @warn "The selected optimization algorithm requires second order derivatives, but `SecondOrder` ADtype was not provided. 
        So a `SecondOrder` with $(prob.f.adtype) for both inner and outer will be created, this can be suboptimal and not work in some cases so 
        an explicit `SecondOrder` ADtype is recommended."
    elseif prob.f.adtype isa AutoZygote &&
           (SciMLBase.requiresconshess(opt) || SciMLBase.requireslagh(opt) ||
            SciMLBase.requireshessian(opt))
        @warn "The selected optimization algorithm requires second order derivatives, but `AutoZygote` ADtype was provided. 
        So a `SecondOrder` with `AutoZygote` for inner and `AutoForwardDiff` for outer will be created, for choosing another pair
        an explicit `SecondOrder` ADtype is recommended."
    end

    f = OptimizationBase.instantiate_function(
        prob.f, reinit_cache, prob.f.adtype, num_cons;
        g = SciMLBase.requiresgradient(opt), h = SciMLBase.requireshessian(opt),
        hv = SciMLBase.requireshessian(opt), fg = SciMLBase.allowsfg(opt),
        fgh = SciMLBase.allowsfgh(opt), cons_j = SciMLBase.requiresconsjac(opt), cons_h = SciMLBase.requiresconshess(opt),
        cons_vjp = SciMLBase.allowsconsjvp(opt), cons_jvp = SciMLBase.allowsconsjvp(opt), lag_h = SciMLBase.requireslagh(opt), sense = prob.sense)

    if structural_analysis
        obj_res, cons_res = symify_cache(f, prob, num_cons, manifold)
    else
        obj_res = nothing
        cons_res = nothing
    end

    return OptimizationCache(f, reinit_cache_passedon, prob.lb, prob.ub, prob.lcons,
        prob.ucons, prob.sense,
        opt, progress, callback, manifold, AnalysisResults(obj_res, cons_res),
        merge((; maxiters, maxtime, abstol, reltol),
            NamedTuple(kwargs)))
end

function SciMLBase.__init(prob::SciMLBase.OptimizationProblem, opt;
        callback = DEFAULT_CALLBACK,
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        progress = false,
        kwargs...)
    return OptimizationCache(prob, opt; maxiters, maxtime, abstol, callback,
        reltol, progress,
        kwargs...)
end

# Wrapper for fields that may change in `reinit!(cache)` of a cache.
mutable struct ReInitCache{uType, P}
    u0::uType
    p::P
end
