function SciMLBase.EnsembleProblem(prob::OptimizationProblem, u0s::Vector{Vector{T}}; kwargs...) where {T}
    prob_func = (prob, i, repeat = nothing) -> remake(prob, u0 = u0s[i])
    return SciMLBase.EnsembleProblem(prob; prob_func, kwargs...)
end

function SciMLBase.EnsembleProblem(prob::OptimizationProblem, trajectories::Int; kwargs...)
    if prob.lb != nothing && prob.ub != nothing
        u0s = QuasiMonteCarlo.sample(trajectories, prob.lb, prob.ub, LatinHypercubeSample())
        prob_func = (prob, i, repeat = nothing) -> remake(prob, u0 = u0s[:, i])
    else
        error("EnsembleProblem requires either initial points as second argument or lower and upper bounds to be defined with the trajectories second argument method.")
    end
    return SciMLBase.EnsembleProblem(prob; prob_func, kwargs...)
end


function SciMLBase.solve(prob::EnsembleProblem{T}, args...; kwargs...) where {T <: OptimizationProblem}
    return SciMLBase.__solve(prob, args...; kwargs...)
end
