abstract type AbstractOptimizationFunction end

struct OptimizationFunction{F,G,H,K} <: AbstractOptimizationFunction
    f::F
    grad::G
    hess::H
    kwargs::K
end

function OptimizationFunction(f; grad=nothing,hess=nothing, p=DiffEqBase.NullParameters(),kwargs...)
    if p isa DiffEqBase.NullParameters
        g! = (res,x) -> ForwardDiff.gradient!(res, f, x)
        h! = (res,x) -> ForwardDiff.hessian!(res, f, x)
    else
        g! = (res,x) -> ForwardDiff.gradient!(res, θ -> f(θ,p), x)
        h! = (res,x) -> ForwardDiff.hessian!(res,  θ -> f(θ,p), x)
    end
    return OptimizationFunction{typeof(f),typeof(g!),typeof(h!),typeof(kwargs)}(f,g!,h!,kwargs)
end
