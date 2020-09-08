abstract type AbstractOptimizationProblem end

struct OptimizationProblem{F,X,P,B,LC,UC,K} <: AbstractOptimizationProblem
    f::F
    x::X
    p::P
    lb::B
    ub::B
    lcons::LC
    ucons::UC
    kwargs::K
    function OptimizationProblem(f, x; p=DiffEqBase.NullParameters(), lb = [], ub = [], lcons = [], ucons = [], kwargs...)
        new{typeof(f), typeof(x), typeof(p), typeof(lb), typeof(lcons), typeof(ucons), typeof(kwargs)}(f, x, p, lb, ub, lcons, ucons, kwargs)
    end
end
