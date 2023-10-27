struct EnsembleOptimizationProblem{T1} <: SciMLBase.AbstractEnsembleProblem
    prob::OptimizationProblem{iip, F, T1} where {iip, F}
    u0s::Vector{T1}
end

function SciMLBase.__init(prob::EnsembleOptimizationProblem{T}, args...; kwargs...) where {T <: OptimizationProblem}
    probs = [remake(prob.prob, u0=u0; kwargs...) for u0 in prob.u0s]
    return [SciMLBase.__init(prob, args...; kwargs...) for prob in probs]
end

function SciMLBase.__solve(caches::Vector{OptimizationCache}, args...; kwargs...)
    return [SciMLBase.__solve(cache, args...; kwargs...) for cache in caches]
end