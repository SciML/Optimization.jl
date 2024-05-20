module OptimizationNLopt

using Reexport
@reexport using NLopt, Optimization
using Optimization.SciMLBase

(f::NLopt.Algorithm)() = f

SciMLBase.allowsbounds(opt::Union{NLopt.Algorithm, NLopt.Opt}) = true
SciMLBase.supports_opt_cache_interface(opt::Union{NLopt.Algorithm, NLopt.Opt}) = true

function SciMLBase.requiresgradient(opt::NLopt.Algorithm) #https://github.com/JuliaOpt/NLopt.jl/blob/master/src/NLopt.jl#L18C7-L18C16
    str_opt = string(opt)
    if str_opt[2] == "D"
        return true
    else
        return false
    end
end

function SciMLBase.requireshessian(opt::NLopt.Algorithm) #https://github.com/JuliaOpt/NLopt.jl/blob/master/src/NLopt.jl#L18C7-L18C16
    str_opt = string(opt)
    if (str_opt[2] == "D" && str_opt[4] == "N")
        return true
    else
        return false
    end
end

function SciMLBase.requireshessian(opt::NLopt.Algorithm) #https://github.com/JuliaOpt/NLopt.jl/blob/master/src/NLopt.jl#L18C7-L18C16
    str_opt = string(opt)
    if str_opt[2] == "D" && str_opt[4] == "N"
        return true
    else
        return false
    end
end
function SciMLBase.requiresconsjac(opt::NLopt.Algorithm) #https://github.com/JuliaOpt/NLopt.jl/blob/master/src/NLopt.jl#L18C7-L18C16
    str_opt = string(opt)
    if str_opt[3] == "O" || str_opt[3] == "I" || str_opt[5] == "G"
        return true
    else
        return false
    end
end

function SciMLBase.requiresgradient(opt::NLopt.Algorithm) #https://github.com/JuliaOpt/NLopt.jl/blob/master/src/NLopt.jl#L18C7-L18C16
    str_opt = string(opt)
    if str_opt[2] == "D"
        return true
    else
        return false
    end
end

function SciMLBase.requireshessian(opt::NLopt.Algorithm) #https://github.com/JuliaOpt/NLopt.jl/blob/master/src/NLopt.jl#L18C7-L18C16
    str_opt = string(opt)
    if (str_opt[2] == "D" && str_opt[4] == "N")
        return true
    else
        return false
    end
end

function SciMLBase.requireshessian(opt::NLopt.Algorithm) #https://github.com/JuliaOpt/NLopt.jl/blob/master/src/NLopt.jl#L18C7-L18C16
    str_opt = string(opt)
    if str_opt[2] == "D" && str_opt[4] == "N"
        return true
    else
        return false
    end
end
function SciMLBase.requiresconsjac(opt::NLopt.Algorithm) #https://github.com/JuliaOpt/NLopt.jl/blob/master/src/NLopt.jl#L18C7-L18C16
    str_opt = string(opt)
    if str_opt[3] == "O" || str_opt[3] == "I" || str_opt[5] == "G"
        return true
    else
        return false
    end
end

function __map_optimizer_args!(cache::OptimizationCache, opt::NLopt.Opt;
        callback = nothing,
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        local_method::Union{NLopt.Algorithm, NLopt.Opt, Nothing} = nothing,
        local_maxiters::Union{Number, Nothing} = nothing,
        local_maxtime::Union{Number, Nothing} = nothing,
        local_options::Union{NamedTuple, Nothing} = nothing,
        kwargs...)
    if local_method !== nothing
        if isa(local_method, NLopt.Opt)
            if ndims(local_method) != length(cache.u0)
                error("Passed local NLopt.Opt optimization dimension does not match OptimizationProblem dimension.")
            end
            local_meth = local_method
        else
            local_meth = NLopt.Opt(local_method, length(cache.u0))
        end

        if !isnothing(local_options)
            for j in Dict(pairs(local_options))
                eval(Meta.parse("NLopt." * string(j.first) * "!"))(local_meth, j.second)
            end
        end

        if !(isnothing(local_maxiters))
            NLopt.maxeval!(local_meth, local_maxiters)
        end

        if !(isnothing(local_maxtime))
            NLopt.maxtime!(local_meth, local_maxtime)
        end

        NLopt.local_optimizer!(opt, local_meth)
    end

    # add optimiser options from kwargs
    for j in kwargs
        eval(Meta.parse("NLopt." * string(j.first) * "!"))(opt, j.second)
    end

    if cache.ub !== nothing
        opt.upper_bounds = cache.ub
    end

    if cache.lb !== nothing
        opt.lower_bounds = cache.lb
    end

    if !(isnothing(maxiters))
        NLopt.maxeval!(opt, maxiters)
    end

    if !(isnothing(maxtime))
        NLopt.maxtime!(opt, maxtime)
    end

    if !isnothing(abstol)
        NLopt.ftol_abs!(opt, abstol)
    end
    if !isnothing(reltol)
        NLopt.ftol_rel!(opt, reltol)
    end

    return nothing
end

function __nlopt_status_to_ReturnCode(status::Symbol)
    if status in Symbol.([
        NLopt.SUCCESS,
        NLopt.STOPVAL_REACHED,
        NLopt.FTOL_REACHED,
        NLopt.XTOL_REACHED,
        NLopt.ROUNDOFF_LIMITED
    ])
        return ReturnCode.Success
    elseif status == Symbol(NLopt.MAXEVAL_REACHED)
        return ReturnCode.MaxIters
    elseif status == Symbol(NLopt.MAXTIME_REACHED)
        return ReturnCode.MaxTime
    elseif status in Symbol.([
        NLopt.OUT_OF_MEMORY,
        NLopt.INVALID_ARGS,
        NLopt.FAILURE,
        NLopt.FORCED_STOP
    ])
        return ReturnCode.Failure
    else
        return ReturnCode.Default
    end
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
        Union{
            NLopt.Algorithm,
            NLopt.Opt
        },
        D,
        P,
        C
}
    local x

    _loss = function (θ)
        x = cache.f(θ, cache.p)
        opt_state = Optimization.OptimizationState(u = θ, objective = x[1])
        if cache.callback(opt_state, x...)
            error("Optimization halted by callback.")
        end
        return x[1]
    end

    fg! = function (θ, G)
        if length(G) > 0
            cache.f.grad(G, θ)
        end

        return _loss(θ)
    end

    opt_setup = if isa(cache.opt, NLopt.Opt)
        if ndims(cache.opt) != length(cache.u0)
            error("Passed NLopt.Opt optimization dimension does not match OptimizationProblem dimension.")
        end
        cache.opt
    else
        NLopt.Opt(cache.opt, length(cache.u0))
    end

    if cache.sense === Optimization.MaxSense
        NLopt.max_objective!(opt_setup, fg!)
    else
        NLopt.min_objective!(opt_setup, fg!)
    end

    maxiters = Optimization._check_and_convert_maxiters(cache.solver_args.maxiters)
    maxtime = Optimization._check_and_convert_maxtime(cache.solver_args.maxtime)

    __map_optimizer_args!(cache, opt_setup; callback = cache.callback, maxiters = maxiters,
        maxtime = maxtime,
        cache.solver_args...)

    t0 = time()
    (minf, minx, ret) = NLopt.optimize(opt_setup, cache.u0)
    t1 = time()
    retcode = __nlopt_status_to_ReturnCode(ret)

    if retcode == ReturnCode.Failure
        @warn "NLopt failed to converge: $(ret)"
    end
    stats = Optimization.OptimizationStats(; time = t1 - t0)
    SciMLBase.build_solution(cache, cache.opt, minx,
        minf; original = opt_setup, retcode = retcode,
        stats = stats)
end

end
