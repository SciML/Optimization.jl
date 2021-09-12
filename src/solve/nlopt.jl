export NLO

struct NLO
    method::Symbol
    NLO(method) = new(method)
end

function __map_optimizer_args(prob::OptimizationProblem, opt::NLopt.Opt; 
    cb=nothing, 
    maxiters::Union{Number, Nothing}=nothing, 
    maxtime::Union{Number, Nothing}=nothing, 
    abstol::Union{Number, Nothing}=nothing, 
    reltol::Union{Number, Nothing}=nothing, 
    local_method::Union{NLO, NLopt.Opt, Nothing} = nothing,
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

    if local_method !== nothing
        if isa(local_method,NLopt.Opt)
            if ndims(local_method) != length(prob.u0)
                error("Passed local NLopt.Opt optimization dimension does not match OptimizationProblem dimension.")
            end
            local_meth = local_method
        else
            local_meth = NLopt.Opt(local_method.method, length(prob.u0))
        end

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

        NLopt.local_optimizer!(opt, local_meth)
    end

    # add optimiser options from kwargs
    for j in kwargs
        eval(Meta.parse("NLopt."*string(j.first)*"!"))(opt, j.second)
    end

    if prob.ub !== nothing
        NLopt.upper_bounds!(opt, prob.ub)
    end

    if prob.lb !== nothing
        NLopt.lower_bounds!(opt, prob.lb)
    end

    if !(isnothing(maxiters))
        NLopt.maxeval!(opt, maxiters)
    end

    if !(isnothing(maxtime))
        NLopt.maxtime!(opt, maxtime)
    end

    return nothing
end


function __solve(prob::OptimizationProblem, opt::Union{NLO, NLopt.Opt};
                 maxiters::Union{Number, Nothing} = nothing,
                 maxtime::Union{Number, Nothing} = nothing,
                 local_method::Union{NLO, NLopt.Opt, Nothing} = nothing,
                 local_maxiters::Union{Number, Nothing} = nothing,
                 local_maxtime::Union{Number, Nothing} = nothing,
                 local_options::Union{NamedTuple,Nothing} = nothing,
                 abstol::Union{Number, Nothing}=nothing,
                 reltol::Union{Number, Nothing}=nothing,
                 progress = false,
                 kwargs...)
    local x

    maxiters = _check_and_convert_maxiters(maxiters)
    maxtime = _check_and_convert_maxtime(maxtime)
    local_maxiters = _check_and_convert_maxiters(local_maxiters)
    local_maxtime = _check_and_convert_maxtime(local_maxtime)


    f = instantiate_function(prob.f,prob.u0,prob.f.adtype,prob.p)

    _loss = function(θ)
        x = f.f(θ, prob.p)
        return x[1]
    end

    fg! = function (θ,G)
        if length(G) > 0
            f.grad(G, θ)
        end

        return _loss(θ)
    end

    if isa(opt,NLopt.Opt)
        if ndims(opt) != length(prob.u0)
            error("Passed NLopt.Opt optimization dimension does not match OptimizationProblem dimension.")
        end
        opt_setup = opt
    else
        opt_setup = NLopt.Opt(opt.method, length(prob.u0))
    end

    NLopt.min_objective!(opt_setup, fg!)

    _map_optimizer_args(prob,opt_setup, maxiters=maxiters, maxtime=maxtime, abstol=abstol, reltol=reltol, local_method=local_method, local_maxiters =local_maxiters, local_options=local_options; kwargs...)

    t0 = time()
    (minf,minx,ret) = NLopt.optimize(opt_setup, prob.u0)
    t1 = time()

    SciMLBase.build_solution(prob, opt, minx, minf; original=opt_setup, retcode=ret)
end
