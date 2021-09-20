function convert_common_kwargs(opt::Union{NonconvexMMA.MMA02, NonconvexMMA.MMA87}, opt_kwargs;
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
        conv_opt_kwargs = (; conv_opt_kwargs..., outer_maxiter=maxiters)
    end

    if !isnothing(maxtime)
        @warn "common maxtime argument is currently not used by $(opt)"
    end

    if !isnothing(abstol)
        tol_tmp = :tol .∈ Ref(keys(conv_opt_kwargs)) ? conv_opt_kwargs[:tol] : Nonconvex.NonconvexCore.Tolerance()
        tol_tmp_args = (; zip(propertynames(tol_tmp), [getproperty(tol_tmp, j) for j in propertynames(tol_tmp)])...)
        tol_tmp_args = (; tol_tmp_args..., fabs = abstol)

        conv_opt_kwargs = (; conv_opt_kwargs..., tol = Nonconvex.NonconvexCore.Tolerance(tol_tmp_args...))
    end
    
    if !isnothing(reltol)
        tol_tmp = :tol .∈ Ref(keys(conv_opt_kwargs)) ? conv_opt_kwargs[:tol] : Nonconvex.NonconvexCore.Tolerance()
        tol_tmp_args = (; zip(propertynames(tol_tmp), [getproperty(tol_tmp, j) for j in propertynames(tol_tmp)])...)
        tol_tmp_args = (; tol_tmp_args..., frel= reltol)

        conv_opt_kwargs = (; conv_opt_kwargs..., tol = Nonconvex.NonconvexCore.Tolerance(tol_tmp_args...))
    end

    return conv_opt_kwargs
end

function _create_options(opt::Union{NonconvexMMA.MMA02, NonconvexMMA.MMA87};
    opt_kwargs=nothing,
    sub_options=nothing,
    convergence_criteria=nothing)

    if !isnothing(convergence_criteria)
        options = (; options = !isnothing(opt_kwargs) ? NonconvexMMA.MAOptions(;opt_kwargs...) : NonconvexMMA.MMAOptions(), convcriteria=convergence_criteria)
    else
        options = (; options = !isnothing(opt_kwargs) ? NonconvexMMA.MMAOptions(;opt_kwargs...) : NonconvexMMA.MMAOptions())
    end

    return options
end

include("nonconvex.jl")