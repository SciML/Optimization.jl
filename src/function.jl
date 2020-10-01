struct AutoForwardDiff <: AbstractADType end
struct AutoReverseDiff <: AbstractADType end
struct AutoTracker <: AbstractADType end
struct AutoZygote <: AbstractADType end
struct AutoFiniteDiff <: AbstractADType end

function OptimizationFunction(f, x, ::AutoForwardDiff,p=DiffEqBase.NullParameters();
                              grad=nothing, hess=nothing, cons = nothing,
                              cons_j = nothing, cons_h = nothing,
                              num_cons = 0, chunksize = 1, hv = nothing)
    _f = θ -> f(θ,p)[1]
    if grad === nothing
        gradcfg = ForwardDiff.GradientConfig(_f, x, ForwardDiff.Chunk{chunksize}())
        grad = (res,θ) -> ForwardDiff.gradient!(res, _f, θ, gradcfg)
    end

    if hess === nothing
        hesscfg = ForwardDiff.HessianConfig(_f, x, ForwardDiff.Chunk{chunksize}())
        hess = (res,θ) -> ForwardDiff.hessian!(res, _f, θ, hesscfg)
    end

    if hv === nothing
        hv = function (H,θ,v)
            res = Array{typeof(x[1])}(undef, length(θ), length(θ)) #DiffResults.HessianResult(θ)
            hess(res, θ)
            H .= res*v
        end
    end

    if cons !== nothing && cons_j === nothing
        if num_cons == 1
            cjconfig = ForwardDiff.JacobianConfig(cons, x, ForwardDiff.Chunk{chunksize}())
            cons_j = (res,θ) -> ForwardDiff.jacobian!(res, cons, θ, cjconfig)
        else
            cons_j = function (res, θ)
                for i in 1:num_cons
                    cjconfig = ForwardDiff.JacobianConfig(x -> cons(x)[i], θ, ForwardDiff.Chunk{chunksize}())
                    ForwardDiff.jacobian!(res[i], x -> cons(x)[i], θ, cjconfig, Val{false}())
                end
            end
        end
    end

    if cons !== nothing && cons_h === nothing
        if num_cons == 1
            cons_h = function (res, θ)
                hess_config_cache = ForwardDiff.HessianConfig(cons, θ, ForwardDiff.Chunk{chunksize}())
                ForwardDiff.hessian!(res, cons, θ, hess_config_cache)
            end
        else
            cons_h = function (res, θ)
                for i in 1:num_cons
                    hess_config_cache = ForwardDiff.HessianConfig(x -> cons(x)[i], θ, ForwardDiff.Chunk{chunksize}())
                    ForwardDiff.hessian!(res[i], x -> cons(x)[i], θ, hess_config_cache, Val{false}())
                end
            end
        end
    end

    return OptimizationFunction{false,AutoForwardDiff,typeof(f),typeof(grad),typeof(hess),typeof(hv),typeof(cons),typeof(cons_j),typeof(cons_h)}(f,AutoForwardDiff(),grad,hess,hv,cons,cons_j,cons_h,num_cons)
end

function OptimizationFunction(f, x, ::AutoZygote, p=DiffEqBase.NullParameters();
                              grad=nothing, hess=nothing, cons = nothing,
                              cons_j = nothing, cons_h = nothing,
                              num_cons = 0, hv = nothing)
    _f = θ -> f(θ,p)[1]
    if grad === nothing
        grad = (res,θ) -> res isa DiffResults.DiffResult ? DiffResults.gradient!(res, Zygote.gradient(_f, θ)[1]) : res .= Zygote.gradient(_f, θ)[1]
    end

    if hess === nothing
        hess = function (res,θ)
            if res isa DiffResults.DiffResult
                DiffResults.hessian!(res, ForwardDiff.jacobian(θ) do θ
                                                Zygote.gradient(_f,θ)[1]
                                            end)
            else
                res .=  ForwardDiff.jacobian(θ) do θ
                    Zygote.gradient(_f,θ)[1]
                  end
            end
        end
    end

    if hv === nothing
        hv = function (H,θ,v)
            _θ = ForwardDiff.Dual.(θ,v)
            res = DiffResults.GradientResult(_θ)
            grad(res,_θ)
            H .= getindex.(ForwardDiff.partials.(DiffResults.gradient(res)),1)
        end
    end
    return OptimizationFunction{false,AutoZygote,typeof(f),typeof(grad),typeof(hess),typeof(hv),typeof(cons),typeof(cons_j),typeof(cons_h)}(f,AutoZygote(),grad,hess,hv,cons,cons_j,cons_h,num_cons)
end

function OptimizationFunction(f, x, ::AutoReverseDiff, p=DiffEqBase.NullParameters();
                              grad=nothing,hess=nothing, cons = nothing,
                              cons_j = nothing, cons_h = nothing,
                              num_cons = 0, hv = nothing)
    _f = θ -> f(θ,p)[1]
    if grad === nothing
        grad = (res,θ) -> ReverseDiff.gradient!(res, _f, θ, ReverseDiff.GradientConfig(θ))
    end

    if hess === nothing
        hess = function (res,θ)
            if res isa DiffResults.DiffResult
                DiffResults.hessian!(res, ForwardDiff.jacobian(θ) do θ
                                                ReverseDiff.gradient(_f,θ)[1]
                                            end)
            else
                res .=  ForwardDiff.jacobian(θ) do θ
                    ReverseDiff.gradient(_f,θ)
                  end
            end
        end
    end

    if hv === nothing
        hv = function (H,θ,v)
            _θ = ForwardDiff.Dual.(θ,v)
            res = DiffResults.GradientResult(_θ)
            grad(res,_θ)
            H .= getindex.(ForwardDiff.partials.(DiffResults.gradient(res)),1)
        end
    end

    return OptimizationFunction{false,AutoReverseDiff,typeof(f),typeof(grad),typeof(hess),typeof(hv),typeof(cons),typeof(cons_j),typeof(cons_h)}(f,AutoReverseDiff(),grad,hess,hv,cons,cons_j,cons_h,num_cons)
end


function OptimizationFunction(f, x, ::AutoTracker, p=DiffEqBase.NullParameters();
                              grad=nothing,hess=nothing, cons = nothing,
                              cons_j = nothing, cons_h = nothing,
                              num_cons = 0, hv = nothing)
    _f = θ -> f(θ,p)[1]
    if grad === nothing
        grad = (res,θ) -> res isa DiffResults.DiffResult ? DiffResults.gradient!(res, Tracker.data(Tracker.gradient(_f, θ)[1])) : res .= Tracker.data(Tracker.gradient(_f, θ)[1])
    end

    if hess === nothing
        hess = (res, θ) -> error("Hessian based methods not supported with Tracker backend, pass in the `hess` kwarg")
    end

    if hv === nothing
        hess = (res, θ) -> error("Hessian based methods not supported with Tracker backend, pass in the `hess` and `hv` kwargs")
    end


    return OptimizationFunction{typeof(f),typeof(grad),typeof(hess),typeof(hv),typeof(cons),typeof(cons_j),typeof(cons_h)}(f,grad,hess,hv,AutoTracker(),cons,cons_j,cons_h,num_cons)
end

function OptimizationFunction(f, x, adtype::AutoFiniteDiff, p=DiffEqBase.NullParameters();
                              grad=nothing,hess=nothing, cons = nothing,
                              cons_j = nothing, cons_h = nothing,
                              num_cons = 0, hv = nothing, fdtype = :forward, fdhtype = :hcentral)
    _f = θ -> f(θ,p)[1]
    if grad === nothing
        grad = (res,θ) -> FiniteDiff.finite_difference_gradient!(res, _f, θ, FiniteDiff.GradientCache(res, x, Val{fdtype}))
    end

    if hess === nothing
        hess = (res,θ) -> FiniteDiff.finite_difference_hessian!(res, _f, θ, FiniteDiff.HessianCache(x, Val{fdhtype}))
    end

    if hv === nothing
        hv = function (H,θ,v)
            res = Array{typeof(x[1])}(undef, length(θ), length(θ)) #DiffResults.HessianResult(θ)
            hess(res, θ)
            H .= res*v
        end
    end

    return OptimizationFunction{false,AutoFiniteDiff,typeof(f),typeof(grad),typeof(hess),typeof(hv),typeof(cons),typeof(cons_j),typeof(cons_h)}(f,adtype,grad,hess,hv,cons,cons_j,cons_h,num_cons)
end
