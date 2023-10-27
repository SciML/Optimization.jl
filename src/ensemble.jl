function SciMLBase.EnsembleProblem(prob::OptimizationProblem, u0s::Vector{Vector{T}}; kwargs...) where {T}
    prob_func = (prob, i, repeat = nothing) -> remake(prob, u0 = u0s[i])
    return SciMLBase.EnsembleProblem(prob; prob_func, kwargs...)
end

function SciMLBase.init(prob::EnsembleProblem{T}, args...; kwargs...) where {T <: OptimizationProblem}
    SciMLBase.__init(prob, args...; kwargs...)
end

function SciMLBase.__init(prob::EnsembleProblem{T}, args...; trajectories,  kwargs...) where {T <: OptimizationProblem}
    return [SciMLBase.__init(prob.prob_func(prob.prob, i), args...; kwargs...) for i in 1:trajectories]
end

function SciMLBase.solve!(cache::Vector{<:OptimizationCache}; kwargs...)
    return [SciMLBase.solve!(cache[i]; kwargs...) for i in eachindex(cache)]
end

function SciMLBase.__solve(caches::Vector{<:OptimizationCache}, args...; kwargs...)
    return [SciMLBase.__solve(cache, args...; kwargs...) for cache in caches]
end