module OptimizationGCMAES

using GCMAES, Optimization, Optimization.SciMLBase

export GCMAESOpt

struct GCMAESOpt end

function __map_optimizer_args(prob::OptimizationProblem, opt::GCMAESOpt;
                              callback = nothing,
                              maxiters::Union{Number, Nothing} = nothing,
                              maxtime::Union{Number, Nothing} = nothing,
                              abstol::Union{Number, Nothing} = nothing,
                              reltol::Union{Number, Nothing} = nothing,
                              kwargs...)

    # add optimiser options from kwargs
    mapped_args = (; kwargs...)

    if !(isnothing(maxiters))
        mapped_args = (; mapped_args..., maxiter = maxiters)
    end

    if !(isnothing(maxtime))
        @warn "common maxtime is currently not used by $(opt)"
    end

    if !isnothing(abstol)
        @warn "common abstol is currently not used by $(opt)"
    end

    if !isnothing(reltol)
        @warn "common reltol is currently not used by $(opt)"
    end

    return mapped_args
end

function SciMLBase.__solve(prob::OptimizationProblem, opt::GCMAESOpt;
                           maxiters::Union{Number, Nothing} = nothing,
                           maxtime::Union{Number, Nothing} = nothing,
                           abstol::Union{Number, Nothing} = nothing,
                           reltol::Union{Number, Nothing} = nothing,
                           progress = false,
                           σ0 = 0.2,
                           kwargs...)
    local x
    local G = similar(prob.u0)

    maxiters = Optimization._check_and_convert_maxiters(maxiters)
    maxtime = Optimization._check_and_convert_maxtime(maxtime)

    f = Optimization.instantiate_function(prob.f, prob.u0, prob.f.adtype, prob.p)

    _loss = function (θ)
        x = f.f(θ, prob.p)
        return x[1]
    end

    if !isnothing(f.grad)
        g = function (θ)
            f.grad(G, θ)
            return G
        end
    end

    opt_args = __map_optimizer_args(prob, opt, maxiters = maxiters, maxtime = maxtime,
                                    abstol = abstol, reltol = reltol; kwargs...)

    t0 = time()
    if prob.sense === Optimization.MaxSense
        opt_xmin, opt_fmin, opt_ret = GCMAES.maximize(isnothing(f.grad) ? _loss :
                                                      (_loss, g), prob.u0, σ0, prob.lb,
                                                      prob.ub; opt_args...)
    else
        opt_xmin, opt_fmin, opt_ret = GCMAES.minimize(isnothing(f.grad) ? _loss :
                                                      (_loss, g), prob.u0, σ0, prob.lb,
                                                      prob.ub; opt_args...)
    end
    t1 = time()

    SciMLBase.build_solution(prob, opt, opt_xmin, opt_fmin; retcode = Symbol(Bool(opt_ret)))
end

end
