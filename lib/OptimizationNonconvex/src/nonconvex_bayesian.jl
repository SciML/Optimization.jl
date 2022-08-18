using NonconvexBayesian
function convert_common_kwargs(opt::NonconvexBayesian.BayesOptAlg, opt_kwargs;
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
        conv_opt_kwargs = (; conv_opt_kwargs..., ftol = reltol)
    end

    return conv_opt_kwargs
end

function __create_options(opt::NonconvexBayesian.BayesOptAlg;
                          opt_kwargs = nothing)
    options = !isnothing(opt_kwargs) ? NonconvexBayesian.BayesOptOptions(; opt_kwargs...) :
              NonconvexBayesian.BayesOptOptions()

    return options
end

function _create_options(opt::NonconvexBayesian.BayesOptAlg;
                         opt_kwargs = nothing,
                         sub_options = nothing,
                         convergence_criteria = nothing)
    options = (;
               options = !isnothing(opt_kwargs) ?
                         NonconvexBayesian.BayesOptOptions(;
                                                           sub_options = __create_options(opt.sub_alg,
                                                                                          opt_kwargs = sub_options),
                                                           opt_kwargs...) :
                         NonconvexBayesian.BayesOptOptions(;
                                                           sub_options = __create_options(alg.sub_alg,
                                                                                          opt_kwargs = sub_options)))

    return options
end

check_optimizer_backend(opt::NonconvexBayesian.BayesOptAlg) = false
