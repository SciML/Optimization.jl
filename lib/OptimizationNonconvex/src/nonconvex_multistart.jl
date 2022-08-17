using NonconvexMultistart
function convert_common_kwargs(opt::NonconvexMultistart.HyperoptAlg, opt_kwargs;
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
        @info "common maxiters argument refers to how many of the potential starting points will be evaluated by $(opt)"
        conv_opt_kwargs = (; conv_opt_kwargs..., iters = maxiters)
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

function __create_options(opt::NonconvexMultistart.HyperoptAlg;
                          opt_kwargs = nothing)
    options = !isnothing(opt_kwargs) ?
              NonconvexMultistart.HyperoptOptions(; opt_kwargs...) :
              NonconvexMultistart.HyperoptOptions()

    if isa(options.sampler, NonconvexMultistart.Hyperopt.Hyperband)
        error("$(options.sampler) is currently not support by Optimization")
    end

    return options
end

function _create_options(opt::NonconvexMultistart.HyperoptAlg;
                         opt_kwargs = nothing,
                         sub_options = nothing,
                         convergence_criteria = nothing)
    options = (;
               options = !isnothing(opt_kwargs) ?
                         NonconvexMultistart.HyperoptOptions(;
                                                             sub_options = __create_options(opt.sub_alg,
                                                                                            opt_kwargs = sub_options),
                                                             opt_kwargs...) :
                         NonconvexMultistart.HyperoptOptions(;
                                                             sub_options = __create_options(opt.sub_alg,
                                                                                            opt_kwargs = sub_options)))

    if isa(options.options.sampler, NonconvexMultistart.Hyperopt.Hyperband)
        error("$(options.options.sampler) is currently not support by Optimization")
    end

    return options
end

check_optimizer_backend(opt::NonconvexMultistart.HyperoptAlg) = false
