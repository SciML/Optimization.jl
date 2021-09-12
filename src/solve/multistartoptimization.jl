function __map_optimizer_args(prob::OptimizationProblem, opt::MultistartOptimization.TikTak;
    cb=nothing, 
    maxiters::Union{Number, Nothing}=nothing, 
    maxtime::Union{Number, Nothing}=nothing, 
    abstol::Union{Number, Nothing}=nothing, 
    reltol::Union{Number, Nothing}=nothing, 
    local_method::Union{NLO, Nothing} = nothing,
    local_maxiters::Union{Number, Nothing} = nothing,
    local_maxtime::Union{Number, Nothing} = nothing,
    local_options::Union{NamedTuple,Nothing} = nothing, 
    kwargs...)

    if !isnothing(abstol)
        @warn "abstol is currently not used by $(opt)"
    end
 
    if !isnothing(reltol)
        @warn "reltol is currently not used by $(opt)"
    end

    if !isnothing(maxtime)
        @warn "maxtime is currently not used by the global method $(opt)"
    end

    if !isnothing(maxiters)
        @warn "maxiters is currently not used by the global method $(opt)"
    end

    if !isnothing(cb)
        @warn "callbacks are currently not used by the global method $(opt)"
    end

    global_meth = opt

    if isa(local_method, NLO)
        local_meth = NLopt.Opt(local_method.method, length(prob.u0))
        
        if !isnothing(local_options)
            for j in Dict(pairs(local_options))
                eval(Meta.parse("NLopt."*string(j.first)*"!"))(local_meth, j.second)
            end
        end

        if !(isnothing(local_maxiters))
            NLopt.maxeval!(local_meth, local_maxiters)
        end

        if !(isnothing(local_maxtime))
            NLopt.maxtime!(local_meth, local_maxtime)
        end
    else
        error("A local method has to be defined using NLO(:algorithm).")
    end

    return global_meth, local_meth
end

function __solve(prob::OptimizationProblem, opt::MultistartOptimization.TikTak;
                maxiters::Union{Number, Nothing}=nothing,
                maxtime::Union{Number, Nothing}=nothing,
                local_method::Union{NLO, Nothing} = nothing,
                local_maxiters::Union{Number, Nothing} = nothing,
                local_maxtime::Union{Number, Nothing} = nothing,
                abstol::Union{Number, Nothing}=nothing,
                reltol::Union{Number, Nothing}=nothing,
                progress = false, kwargs...)
    local x, _loss

    local_maxiters = _check_and_convert_maxiters(local_maxiters)
    local_maxtime = _check_and_convert_maxtime(local_maxtime)


    _loss = function(θ)
        x = prob.f(θ, prob.p)
        return first(x)
    end

    opt_setup = MultistartOptimization.MinimizationProblem(_loss, prob.lb, prob.ub)
    multistart_method, local_method = _map_optimizer_args(prob, opt, maxiters=maxiters, abstol=abstol, reltol=reltol,  local_method=local_method, local_maxiters =local_maxiters, local_maxtime=local_maxtime, local_options=local_options; kwargs...)

    t0 = time()
    opt_res = MultistartOptimization.multistart_minimization(multistart_method, local_method, opt_setup)
    t1 = time()
    opt_ret = opt_res.ret

    SciMLBase.build_solution(prob, opt, p.location, p.value; original=p, retcode=opt_ret)
end
