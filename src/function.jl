abstract type AbstractOptimizationFunction end
abstract type AbstractADType end

struct AutoForwardDiff <: AbstractADType end
struct AutoReverseDiff <: AbstractADType end
struct AutoTracker <: AbstractADType end
struct AutoZygote <: AbstractADType end
struct AutoFiniteDiff <: AbstractADType end
struct AutoModelingToolkit <: AbstractADType end

struct OptimizationFunction{F,G,H,HV,K} <: AbstractOptimizationFunction
    f::F
    grad::G
    hess::H
    hv::HV
    adtype::AbstractADType
    kwargs::K
end

function OptimizationFunction(f, x, ::AutoForwardDiff; grad=nothing,hess=nothing, p=DiffEqBase.NullParameters(), chunksize = 1, hv = nothing, kwargs...)
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

    return OptimizationFunction{typeof(f),typeof(grad),typeof(hess),typeof(hv),typeof(kwargs)}(f,grad,hess,hv,AutoForwardDiff(),kwargs)
end

function OptimizationFunction(f, x, ::AutoZygote; grad=nothing, hess=nothing, p=DiffEqBase.NullParameters(), hv = nothing, kwargs...)
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
    return OptimizationFunction{typeof(f),typeof(grad),typeof(hess),typeof(hv),typeof(kwargs)}(f,grad,hess,hv,AutoZygote(),kwargs)
end

function OptimizationFunction(f, x, ::AutoReverseDiff; grad=nothing,hess=nothing, p=DiffEqBase.NullParameters(), hv = nothing, kwargs...)
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

    return OptimizationFunction{typeof(f),typeof(grad),typeof(hess),typeof(hv),typeof(kwargs)}(f,grad,hess,hv,AutoReverseDiff(),kwargs)
end


function OptimizationFunction(f, x, ::AutoTracker; grad=nothing,hess=nothing, p=DiffEqBase.NullParameters(), hv = nothing, kwargs...)
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


    return OptimizationFunction{typeof(f),typeof(grad),typeof(hess),typeof(hv),typeof(kwargs)}(f,grad,hess,hv,AutoTracker(),kwargs)
end

function OptimizationFunction(f, x, adtype::AutoFiniteDiff; grad=nothing,hess=nothing, p=DiffEqBase.NullParameters(), hv = nothing, fdtype = :forward, fdhtype = :hcentral, kwargs...)
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

    return OptimizationFunction{typeof(f),typeof(grad),typeof(hess),typeof(hv),typeof(kwargs)}(f,grad,hess,hv,adtype,kwargs)
end
