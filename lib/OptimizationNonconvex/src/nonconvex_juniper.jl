using NonconvexJuniper
function convert_common_kwargs(opt::NonconvexJuniper.JuniperIpoptAlg, opt_kwargs;
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
        @warn "common maxiters argument is currently not used by $(opt)"
    end

    if !isnothing(maxtime)
        conv_opt_kwargs = (; conv_opt_kwargs..., time_limit = maxtime)
    end

    if !isnothing(abstol)
        conv_opt_kwargs = (; conv_opt_kwargs..., atol = abstol)
    end

    if !isnothing(reltol)
        @warn "common reltol argument is currently not used by $(opt)"
    end

    return conv_opt_kwargs
end

function _create_options(opt::NonconvexJuniper.JuniperIpoptAlg;
                         opt_kwargs = nothing,
                         sub_options = nothing,
                         convergence_criteria = nothing)
    if !isnothing(sub_options)
        options = (;
                   options = !isnothing(opt_kwargs) ?
                             NonconvexJuniper.JuniperIpoptOptions(;
                                                                  subsolver_options = IpoptOptions(sub_options...),
                                                                  opt_kwargs...) :
                             NonconvexJuniper.JuniperIpoptOptions(subsolver_options = IpoptOptions(sub_options...)))
    else
        options = (;
                   options = !isnothing(opt_kwargs) ?
                             NonconvexJuniper.JuniperIpoptOptions(;
                                                                  subsolver_options = IpoptOptions(),
                                                                  opt_kwargs...) :
                             NonconvexJuniper.JuniperIpoptOptions(subsolver_options = IpoptOptions()))
    end
    return options
end

check_optimizer_backend(opt::NonconvexJuniper.JuniperIpoptAlg) = false
