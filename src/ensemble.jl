function SciMLBase.EnsembleProblem(prob::OptimizationProblem, u0s::Vector{Vector{T}}; kwargs...) where {T}
    prob_func = (prob, i, repeat = nothing) -> remake(prob, u0 = u0s[i])
    return SciMLBase.EnsembleProblem(prob; prob_func, kwargs...)
end


function SciMLBase.solve(prob::EnsembleProblem{T}, args...; kwargs...) where {T <: OptimizationProblem}
    return SciMLBase.__solve(prob, args...; kwargs...)
end
