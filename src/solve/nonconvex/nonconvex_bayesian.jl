function convert_common_kwargs(opt::NonconvexBayesian.BayesOptAlg, opt_kwargs;
    cb=nothing,
    maxiters=nothing,
    maxtime=nothing,
    abstol=nothing,
    reltol=nothing)

    conv_opt_kwargs = (; opt_kwargs...)

    if !isnothing(cb)
        @warn "common callback argument is currently not used by $(opt)"
    end
  
    if !isnothing(maxiters)
        conv_opt_kwargs = (; conv_opt_kwargs..., maxiter=maxiters)
    end

    if !isnothing(maxtime)
        @warn "common maxtime argument is currently not used by $(opt)"
    end

    if !isnothing(abstol)
        @warn "common abstol argument is currently not used by $(opt)"
    end
    
    if !isnothing(reltol)
        conv_opt_kwargs = (; conv_opt_kwargs..., ftol =reltol)
    end

    return conv_opt_kwargs
end

function _create_options(opt::NonconvexBayesian.BayesOptAlg;
    opt_kwargs=nothing,
    sub_options=nothing,
    convergence_criteria=nothing)

    if !isnothing(sub_options)
        options = (; options = !isnothing(opt_kwargs) ? BayesOptOptions(;sub_options= _create_options(opt.sub_alg, opt_kwargs=sub_options) ,kwargs...) : BayesOptOptions(;sub_options= _create_options(alg.sub_alg,opt_kwargs= sub_options)))
    else    
        options = (; options = !isnothing(opt_kwargs) ? BayesOptOptions(;opt_kwargs...) : BayesOptOptions())
    end
    
    return options
end

include("nonconvex.jl")