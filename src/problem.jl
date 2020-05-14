abstract type AbstractOptimizationProblem end

struct OptimizationProblem{F,U,P,B,K} <: AbstractOptimizationProblem
    f::F
    u0::U
    p::P
    lb::B
    ub::B
    kwargs::K
    function OptimizationProblem(f, p, u0=nothing; lb = nothing, ub = nothing, kwargs...)
        new{typeof(f), typeof(u0), typeof(p), typeof(lb), typeof(kwargs)}(f, u0, p, lb, ub, kwargs)
    end
end
