function convert_common_kwargs(opt::NonconvexMultistart.HyperoptAlg, opt_kwargs;
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
        @warn "common maxiters argument is currently not used by $(opt)"
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

function _create_options(opt::NonconvexMultistart.HyperoptAlg;
    opt_kwargs=nothing,
    sub_options=nothing,
    convergence_criteria=nothing)

    if !isnothing(sub_options)
        options = (; options = !isnothing(opt_kwargs) ? NonconvexMultistart.HyperoptOptions(;sub_options= _create_options(opt.sub_alg, sub_options) ,kwargs...) : NonconvexMultistart.HyperoptOptions(;sub_options= _create_options(opt.sub_alg, sub_options)))
    else
        options =  (; options = !isnothing(opt_kwargs) ? NonconvexMultistart.HyperoptOptions(;opt_kwargs...) : NonconvexMultistart.HyperoptOptions())
    end
    if isa(options.sampler, Hyperband)
        error("$(options.sampler) is currently not support by GalacticOptim")
    end
    
    return options
end

include("nonconvex.jl")