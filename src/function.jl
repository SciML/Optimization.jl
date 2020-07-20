abstract type AbstractOptimizationFunction end
abstract type AbstractADType end

struct AutoForwardDiff <: AbstractADType end
struct AutoReverseDiff <: AbstractADType end
struct AutoTracker <: AbstractADType end
struct AutoZygote <: AbstractADType end
struct AutoFiniteDiff <: AbstractADType end
struct AutoModelingToolkit <: AbstractADType end

struct OptimizationFunction{F,G,H,K} <: AbstractOptimizationFunction
    f::F
    grad::G
    hess::H
    adtype::AbstractADType
    kwargs::K
end

function OptimizationFunction(f, x, ::AutoForwardDiff; grad=nothing,hess=nothing, p=DiffEqBase.NullParameters(), chunksize = 1, kwargs...)
    _f = θ -> f(θ,p)[1]
    if grad === nothing
        cfg = ForwardDiff.GradientConfig(_f, x, Chunk{chunksize}())
        grad = (res,θ) -> ForwardDiff.gradient!(res, _f, θ, cfg)
    end

    if hess === nothing
        cfg = ForwardDiff.HessiantConfig(_f, x, Chunk{chunksize}())
        hess = (res,θ) -> ForwardDiff.hessian!(res, _f, θ, cfg)
    end
    return OptimizationFunction{typeof(f),typeof(grad),typeof(hess),typeof(kwargs)}(f,grad,hess,AutoForwardDiff(),kwargs)
end

function OptimizationFunction(f, x, ::AutoZygote; grad=nothing,hess=nothing, p=DiffEqBase.NullParameters(),kwargs...)
    _f = θ -> f(θ,p)[1]
    if grad === nothing
        grad = (res,θ) -> res isa DiffResults.DiffResult ? DiffResults.gradient!(res, Zygote.gradient(_f, θ)[1]) : res .= Zygote.gradient(_f, θ)[1]
    end

    if hess === nothing
        hess = (res,θ) -> res isa DiffResults.DiffResult ? DiffResults.hessian!(res, Zygote.hessian(_f, θ)) : res .= Zygote.hessian(_f, θ)
    end
    return OptimizationFunction{typeof(f),typeof(grad),typeof(hess),typeof(kwargs)}(f,grad,hess,AutoZygote(),kwargs)
end