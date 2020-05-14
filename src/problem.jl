abstract type AbstractOptimizationProblem end

struct OptimizationProblem{F,X,P,B,K} <: AbstractOptimizationProblem
    f::F
    x::X
    p::P
    lb::B
    ub::B
    kwargs::K
    function OptimizationProblem(f, x; p=DiffEqBase.NullParameters(), lb = nothing, ub = nothing, kwargs...)
        new{typeof(f), typeof(x), typeof(p), typeof(lb), typeof(kwargs)}(f, x, p, lb, ub, kwargs)
    end
end
