function convert_common_kwargs(opt::NonconvexNLopt.NLoptAlg, opt_kwargs;
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
        conv_opt_kwargs = (; conv_opt_kwargs..., maxeval=maxiters)
     end

    if !isnothing(maxtime)
        conv_opt_kwargs = (; conv_opt_kwargs..., maxtime = maxtime)
    end

    if !isnothing(abstol)
        conv_opt_kwargs = (; conv_opt_kwargs..., ftol_abs = abstol)
    end
    
    if !isnothing(reltol)
        conv_opt_kwargs = (; conv_opt_kwargs..., ftol_rel =reltol)
    end

    return conv_opt_kwargs
end

function _create_options(opt::NonconvexNLopt.NLoptAlg;
    opt_kwargs=nothing,
    sub_options=nothing,
    convergence_criteria=nothing)

    options = (; options = !isnothing(opt_kwargs) ? NonconvexNLopt.NLoptOptions(;opt_kwargs...) : NonconvexNLopt.NLoptOptions())

    return options
end

include("nonconvex.jl")