abstract type AbstractOptimizationFunction end

struct OptimizationFunction{F,G,H,K} <: AbstractOptimizationFunction
    f::F
    grad::G
    hes::H
    kwargs::K
end

function OptimizationFunction(f,u0=nothing;grad=nothing,hes=nothing,kwargs...)
    if u0 !== nothing
        f_ = x -> f(u0,x)
    else
        f_ = f
    end
    g! = (res,x) -> ForwardDiff.gradient!(res, f_, x)
    h! = (res,x) -> ForwardDiff.hessian!(res, f_, x)
    return OptimizationFunction{typeof(f),typeof(g!),typeof(h!),typeof(kwargs)}(f,g!,h!,kwargs)
end
