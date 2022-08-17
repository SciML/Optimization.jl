using NonconvexMMA
function convert_common_kwargs(opt::Union{NonconvexMMA.MMA02, NonconvexMMA.MMA87},
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
        conv_opt_kwargs = (; conv_opt_kwargs..., outer_maxiter = maxiters)
    end

    if !isnothing(maxtime)
        @warn "common maxtime argument is currently not used by $(opt)"
    end

    if !isnothing(abstol)
        tol_tmp = :tol .∈ Ref(keys(conv_opt_kwargs)) ? conv_opt_kwargs[:tol] :
                  Nonconvex.NonconvexCore.Tolerance()
        tol_tmp_args = (;
                        zip(propertynames(tol_tmp),
                            [getproperty(tol_tmp, j) for j in propertynames(tol_tmp)])...)
        tol_tmp_args = (; tol_tmp_args..., fabs = abstol)

        conv_opt_kwargs = (; conv_opt_kwargs...,
                           tol = Nonconvex.NonconvexCore.Tolerance(tol_tmp_args...))
    end

    if !isnothing(reltol)
        tol_tmp = :tol .∈ Ref(keys(conv_opt_kwargs)) ? conv_opt_kwargs[:tol] :
                  Nonconvex.NonconvexCore.Tolerance()
        tol_tmp_args = (;
                        zip(propertynames(tol_tmp),
                            [getproperty(tol_tmp, j) for j in propertynames(tol_tmp)])...)
        tol_tmp_args = (; tol_tmp_args..., frel = reltol)

        conv_opt_kwargs = (; conv_opt_kwargs...,
                           tol = Nonconvex.NonconvexCore.Tolerance(tol_tmp_args...))
    end

    return conv_opt_kwargs
end

function __create_options(opt::Union{NonconvexMMA.MMA02, NonconvexMMA.MMA87};
                          opt_kwargs = nothing)
    options = !isnothing(opt_kwargs) ? NonconvexMMA.MMAOptions(; opt_kwargs...) :
              NonconvexMMA.MMAOptions()

    return options
end

function _create_options(opt::Union{NonconvexMMA.MMA02, NonconvexMMA.MMA87};
                         opt_kwargs = nothing,
                         sub_options = nothing,
                         convergence_criteria = nothing)
    if !isnothing(convergence_criteria)
        options = (; options = __create_options(opt, opt_kwargs = opt_kwargs),
                   convcriteria = convergence_criteria)
    else
        options = (; options = __create_options(opt, opt_kwargs = opt_kwargs))
    end

    return options
end

check_optimizer_backend(opt::Union{NonconvexMMA.MMA02, NonconvexMMA.MMA87}) = false
