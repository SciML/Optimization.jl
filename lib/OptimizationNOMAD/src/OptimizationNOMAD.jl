module OptimizationNOMAD

using NOMAD, Optimization, Optimization.SciMLBase

export NOMADOpt
struct NOMADOpt end

function __map_optimizer_args!(prob::OptimizationProblem, opt::NOMAD.NomadProblem;
                               callback = nothing,
                               maxiters::Union{Number, Nothing} = nothing,
                               maxtime::Union{Number, Nothing} = nothing,
                               abstol::Union{Number, Nothing} = nothing,
                               reltol::Union{Number, Nothing} = nothing,
                               kwargs...)
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
        @warn "common reltol is currently not used by $(opt)"
    end

    if !isnothing(abstol)
        @warn "common abstol is currently not used by $(opt)"
    end

    return nothing
end

function SciMLBase.__solve(prob::OptimizationProblem, opt::NOMADOpt;
                           maxiters::Union{Number, Nothing} = nothing,
                           maxtime::Union{Number, Nothing} = nothing,
                           abstol::Union{Number, Nothing} = nothing,
                           reltol::Union{Number, Nothing} = nothing,
                           progress = false,
                           kwargs...)
    local x

    maxiters = Optimization._check_and_convert_maxiters(maxiters)
    maxtime = Optimization._check_and_convert_maxtime(maxtime)

    _loss = function (θ)
        x = prob.f(θ, prob.p)
        return first(x)
    end

    function bb(x)
        l = _loss(x)
        success = !isnan(l) && !isinf(l)
        count_eval = true
        return (success, count_eval, [l])
    end

    if !isnothing(prob.lcons) | !isnothing(prob.ucons)
        @warn "Linear and nonlinear constraints defined in OptimizationProblem are currently not used by $(opt)"
    end

    bounds = (;)
    if !isnothing(prob.lb)
        bounds = (; bounds..., lower_bound = prob.lb)
    end

    if !isnothing(prob.ub)
        bounds = (; bounds..., upper_bound = prob.ub)
    end

    opt_setup = NOMAD.NomadProblem(length(prob.u0), 1, ["OBJ"], bb; bounds...)

    __map_optimizer_args!(prob, opt_setup, maxiters = maxiters, maxtime = maxtime,
                          abstol = abstol, reltol = reltol; kwargs...)

    t0 = time()
    opt_res = NOMAD.solve(opt_setup, prob.u0)
    t1 = time()

    SciMLBase.build_solution(prob, opt, opt_res.x_best_feas, first(opt_res.bbo_best_feas);
                             original = opt_res)
end

end
