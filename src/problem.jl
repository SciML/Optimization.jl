abstract type AbstractOptimizationProblem end

struct OptimizationProblem{F,U,P,B} <: AbstractOptimizationProblem
    f::F
    u0::U
    p::P
    lb::B
    ub::B
    function OptimizationProblem(f, p, u0=nothing,; lb = nothing, ub = nothing)
        new{typeof(f), typeof(u0), typeof(p), typeof(lb)}(f, u0, p, lb, ub)
    end
end
