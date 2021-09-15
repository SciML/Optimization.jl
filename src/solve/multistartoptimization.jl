function __map_optimizer_args(prob::OptimizationProblem, opt::MultistartOptimization.TikTak;
    cb=nothing, 
    maxiters::Union{Number, Nothing}=nothing, 
    maxtime::Union{Number, Nothing}=nothing, 
    abstol::Union{Number, Nothing}=nothing, 
    reltol::Union{Number, Nothing}=nothing, 
    local_method::Union{NLopt.Algorithm, Nothing} = nothing,
    local_maxiters::Union{Number, Nothing} = nothing,
    local_maxtime::Union{Number, Nothing} = nothing,
    local_options::Union{NamedTuple,Nothing} = nothing, 
    kwargs...)

    if !isnothing(abstol)
        @warn "abstol is currently not used by the global method $(opt). Set the absolute tolerance of the optimized states for the local method via local_options using the keyword xtol_abs"
    end
 
    if !isnothing(reltol)
        @warn "reltol is currently not used by the global method $(opt). Set the relative tolerance of the optimized states for the local method via local_options using the keyword xtol_rel"
    end

    if !isnothing(maxtime)
        @warn "maxtime is currently not used by the global method $(opt). Set maxtime of local optimiser via local_maxtime."
    end

    if !isnothing(maxiters)
        @warn "maxiters is currently not used by the global method $(opt). Set maxiters of local optimiser via local_maxiters."
    end

    if !isnothing(cb)
        @warn "callbacks are currently not used by the global method $(opt)"
    end

    global_meth = opt

    if isa(local_method, NLopt.Algorithm)
        local_kwargs = (; )
        if !(isnothing(local_maxiters))
            local_kwargs = (; local_kwargs..., maxeval=local_maxiters)
        end

        if !(isnothing(local_maxtime))
            local_kwargs = (; local_kwargs..., maxtime=local_maxtime)
        end

        if !(isnothing(local_options))
            local_kwargs = (; local_kwargs..., local_options...)
        end

        local_meth = MultistartOptimization.NLoptLocalMethod(local_method; local_kwargs...)
    else
        error("A local method has to be defined using algorithm of type NLopt.Algorithm e.g. 'NLopt.LN_NELDERMEAD()'.")
    end

    return global_meth, local_meth
end

function __solve(prob::OptimizationProblem, opt::MultistartOptimization.TikTak;
                maxiters::Union{Number, Nothing}=nothing,
                maxtime::Union{Number, Nothing}=nothing,
                abstol::Union{Number, Nothing}=nothing,
                reltol::Union{Number, Nothing}=nothing,
                local_method::Union{NLopt.Algorithm, Nothing} = nothing,
                local_maxiters::Union{Number, Nothing} = nothing,
                local_maxtime::Union{Number, Nothing} = nothing,
                local_options::Union{NamedTuple,Nothing} = nothing,
                progress = false, kwargs...)
    local x, _loss

    maxiters = _check_and_convert_maxiters(maxiters)
    maxtime = _check_and_convert_maxtime(maxtime)

    local_maxiters = _check_and_convert_maxiters(local_maxiters)
    local_maxtime = _check_and_convert_maxtime(local_maxtime)


    _loss = function(θ)
        x = prob.f(θ, prob.p)
        return first(x)
    end

    opt_setup = MultistartOptimization.MinimizationProblem(_loss, prob.lb, prob.ub)
    multistart_method, local_method = _map_optimizer_args(prob, opt, maxiters=maxiters, maxtime=maxtime, abstol=abstol, reltol=reltol,  local_method=local_method, local_maxiters =local_maxiters, local_maxtime=local_maxtime, local_options=local_options; kwargs...)

    t0 = time()
    opt_res = MultistartOptimization.multistart_minimization(multistart_method, local_method, opt_setup)
    t1 = time()
    opt_ret = hasproperty(opt_res, :ret) ? opt_res.ret : nothing

    SciMLBase.build_solution(prob, opt, opt_res.location, opt_res.value; (isnothing(opt_ret) ? (; original=opt_res) : (; original=opt_res, retcode=opt_ret))... )
end
