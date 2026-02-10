module OptimizationNOMAD

using Reexport
@reexport using OptimizationBase
using NOMAD, SciMLBase

export NOMADOpt
struct NOMADOpt end

@enum ConstraintBarrierType ExtremeBarrierMethod ProgressiveBarrierMethod

SciMLBase.allowsbounds(::NOMADOpt) = true
SciMLBase.allowscallback(::NOMADOpt) = false
SciMLBase.allowsconstraints(::NOMADOpt) = true

function __map_optimizer_args!(
        prob::OptimizationProblem, opt::NOMAD.NomadProblem;
        callback = nothing,
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        verbose = OptimizationBase.OptimizationVerbosity(),
        kwargs...
    )
    for j in kwargs
        setproperty!(opt.options, j.first, j.second)
    end

    if !isnothing(maxiters)
        opt.options.max_bb_eval = maxiters
    end

    if !isnothing(maxtime)
        opt.options.max_time = maxtime
    end

    if !isnothing(reltol)
        @SciMLMessage(
            lazy"common reltol is currently not used by $(opt)",
            verbose, :unsupported_kwargs
        )
    end

    if !isnothing(abstol)
        @SciMLMessage(
            lazy"common abstol is currently not used by $(opt)",
            verbose, :unsupported_kwargs
        )
    end

    return nothing
end

@inline strcnsmethod(m::ConstraintBarrierType) = m === ExtremeBarrierMethod ? "EB" : "PB"

function SciMLBase.__solve(
        prob::OptimizationProblem, opt::NOMADOpt;
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        cons_method = ExtremeBarrierMethod,
        verbose = OptimizationBase.OptimizationVerbosity(),
        kwargs...
    )
    local x

    maxiters = OptimizationBase._check_and_convert_maxiters(maxiters)
    maxtime = OptimizationBase._check_and_convert_maxtime(maxtime)

    _loss = function (θ)
        x = prob.f(θ, prob.p)
        return first(x)
    end

    if prob.f.cons === nothing
        function bb(x)
            l = _loss(x)
            success = !isnan(l) && !isinf(l)
            count_eval = true
            return (success, count_eval, [l])
        end
    else
        eqinds = findall(i -> prob.lcons[i] == prob.ucons[i], 1:length(prob.ucons))
        function bbcons(x)
            l = _loss(x)
            c = zeros(eltype(x), length(prob.ucons))
            prob.f.cons(c, x, prob.p)
            c -= prob.ucons
            if !isempty(eqinds)
                c[eqinds] = abs.(c[eqinds])
            end
            success = !isnan(l) && !isinf(l)
            count_eval = true
            return (success, count_eval, vcat(l, c))
        end
    end

    bounds = (;)
    if !isnothing(prob.lb)
        bounds = (; bounds..., lower_bound = prob.lb)
    end

    if !isnothing(prob.ub)
        bounds = (; bounds..., upper_bound = prob.ub)
    end

    if prob.f.cons === nothing
        opt_setup = NOMAD.NomadProblem(length(prob.u0), 1, ["OBJ"], bb; bounds...)
    else
        opt_setup = NOMAD.NomadProblem(
            length(prob.u0), 1 + length(prob.ucons),
            vcat("OBJ", fill(strcnsmethod(cons_method), length(prob.ucons))),
            bbcons; bounds...
        )
    end

    __map_optimizer_args!(
        prob, opt_setup, maxiters = maxiters, maxtime = maxtime,
        abstol = abstol, reltol = reltol, verbose = verbose; kwargs...
    )

    t0 = time()
    opt_res = NOMAD.solve(opt_setup, prob.u0)
    t1 = time()
    stats = OptimizationBase.OptimizationStats(; time = t1 - t0)
    return SciMLBase.build_solution(
        SciMLBase.DefaultOptimizationCache(prob.f, prob.p), opt,
        opt_res.x_best_feas, first(opt_res.bbo_best_feas);
        original = opt_res, stats = stats
    )
end

end
