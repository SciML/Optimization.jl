using NonconvexSearch
function convert_common_kwargs(opt::Union{NonconvexSearch.MTSAlg, NonconvexSearch.LS1Alg},
                               opt_kwargs;
                               callback = nothing,
                               maxiters = nothing,
                               maxtime = nothing,
                               abstol = nothing,
                               reltol = nothing)
    conv_opt_kwargs = (; opt_kwargs...)

    if !isnothing(callback)
        @warn "common callback argument is currently not used by $(opt)"
    end

    if !isnothing(maxiters)
        conv_opt_kwargs = (; conv_opt_kwargs..., maxiter = maxiters)
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

function __create_options(opt::Union{NonconvexSearch.MTSAlg, NonconvexSearch.LS1Alg};
                          opt_kwargs = nothing)
    options = !isnothing(opt_kwargs) ? NonconvexSearch.MTSOptions(; opt_kwargs...) :
              NonconvexSearch.MTSOptions()

    return options
end

function _create_options(opt::Union{NonconvexSearch.MTSAlg, NonconvexSearch.LS1Alg};
                         opt_kwargs = nothing,
                         sub_options = nothing,
                         convergence_criteria = nothing)
    options = (; options = __create_options(opt, opt_kwargs = opt_kwargs))

    return options
end

check_optimizer_backend(opt::Union{NonconvexSearch.MTSAlg, NonconvexSearch.LS1Alg}) = true
