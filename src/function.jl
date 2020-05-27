abstract type AbstractOptimizationFunction end

struct OptimizationFunction{F,G,H,K} <: AbstractOptimizationFunction
    f::F
    grad::G
    hess::H
    kwargs::K
end

function OptimizationFunction(f,x; grad=nothing,hess=nothing, p=DiffEqBase.NullParameters(),kwargs...)
    _f = θ -> f(θ,p)
    g! = (res,θ) -> ForwardDiff.gradient!(res, _f, θ)
    h! = (res,θ) -> ForwardDiff.hessian!(res, _f, θ)
    return OptimizationFunction{typeof(f),typeof(g!),typeof(h!),typeof(kwargs)}(f,g!,h!,kwargs)
end
