module OptimizationSpeedMapping

using SpeedMapping, Optimization, Optimization.SciMLBase

export SpeedMappingOpt

struct SpeedMappingOpt end

function __map_optimizer_args(prob::OptimizationProblem, opt::SpeedMappingOpt;
                              callback = nothing,
                              maxiters::Union{Number, Nothing} = nothing,
                              maxtime::Union{Number, Nothing} = nothing,
                              abstol::Union{Number, Nothing} = nothing,
                              reltol::Union{Number, Nothing} = nothing,
                              kwargs...)

    # add optimiser options from kwargs
    mapped_args = (; kwargs...)

    if !(isnothing(maxiters))
        @info "maxiters defines maximum gradient calls for $(opt)"
        mapped_args = (; mapped_args..., maps_limit = maxiters)
    end

    if !(isnothing(maxtime))
        mapped_args = (; mapped_args..., time_limit = maxtime)
    end

    if !isnothing(abstol)
        @warn "common abstol is currently not used by $(opt)"
    end

    if !isnothing(reltol)
        @warn "common reltol is currently not used by $(opt)"
    end

    return mapped_args
end

function SciMLBase.__solve(prob::OptimizationProblem, opt::SpeedMappingOpt;
                           maxiters::Union{Number, Nothing} = nothing,
                           maxtime::Union{Number, Nothing} = nothing,
                           abstol::Union{Number, Nothing} = nothing,
                           reltol::Union{Number, Nothing} = nothing,
                           progress = false,
                           kwargs...)
    local x

    maxiters = Optimization._check_and_convert_maxiters(maxiters)
    maxtime = Optimization._check_and_convert_maxtime(maxtime)

    f = Optimization.instantiate_function(prob.f, prob.u0, prob.f.adtype, prob.p)

    _loss = function (θ)
        x = f.f(θ, prob.p)
        return first(x)
    end

    if isnothing(f.grad)
        @info "SpeedMapping's ForwardDiff AD backend is used to calculate the gradient information."
    end

    opt_args = __map_optimizer_args(prob, opt, maxiters = maxiters, maxtime = maxtime,
                                    abstol = abstol, reltol = reltol; kwargs...)

    t0 = time()
    opt_res = SpeedMapping.speedmapping(prob.u0; f = _loss, (g!) = f.grad, lower = prob.lb,
                                        upper = prob.ub, opt_args...)
    t1 = time()
    opt_ret = Symbol(opt_res.converged)

    SciMLBase.build_solution(prob, opt, opt_res.minimizer, _loss(opt_res.minimizer);
                             original = opt_res, retcode = opt_ret)
end

end
