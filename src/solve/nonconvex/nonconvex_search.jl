function convert_common_kwargs(opt::Union{NonconvexSearch.MTSAlg, NonconvexSearch.LS1Alg}, opt_kwargs;
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
        @warn "common reltol argument is currently not used by $(opt)"
    end

    return conv_opt_kwargs
end

function _create_options(opt::Union{NonconvexSearch.MTSAlg, NonconvexSearch.LS1Alg};
    opt_kwargs=nothing,
    sub_options=nothing,
    convergence_criteria=nothing)

    options = (; options = !isnothing(opt_kwargs) ? NonconvexSearch.MTSOptions(;opt_kwargs...) : NonconvexSearch.MTSOptions())
    
    return options
end

include("nonconvex.jl")