module OptimizationAccelerated
using Optimization, Optimization.SciMLBase, LinearAlgebra
using OptimizationMOI, OSQP, ModelingToolkit

struct AcceleratedOpt
    β::Float64
    δ::Float64
    Tₖ::Float64
    α::Float64
    ϵ::Float64
    ϵ_sol::Float64
    ϵ_const::Float64
end

SciMLBase.supports_opt_cache_interface(opt::AcceleratedOpt) = true
SciMLBase.allowsconstraints(::AcceleratedOpt) = true
SciMLBase.requiresconstraints(::AcceleratedOpt) = true

function SciMLBase.__solve(cache::OptimizationCache{
    F,
    RC,
    LB,
    UB,
    LC,
    UC,
    S,
    O,
    D,
    P,
    C,
}) where {
    F,
    RC,
    LB,
    UB,
    LC,
    UC,
    S,
    O <: AcceleratedOpt,
    D,
    P,
    C,
}
    β, δ, Tₖ, α, ϵ, ϵ_sol, ϵ_const = cache.opt.β, cache.opt.δ, cache.opt.Tₖ, cache.opt.α, cache.opt.ϵ, cache.opt.ϵ_sol, cache.opt.ϵ_const

    xₖ = cache.u0
    uₖ = zeros(eltype(xₖ), length(xₖ))
    gradcache = zeros(eltype(xₖ), length(xₖ))
    conscache = zeros(eltype(xₖ), length(cache.lcons))
    consjaccache = zeros(eltype(xₖ), length(cache.lcons), length(xₖ))
    for i in 1:cache.solver_args.maxiters
        gradcache .= zero(eltype(xₖ))
        cache.f.grad(gradcache, xₖ + β*uₖ)
        rₖ = uₖ - 2*δ*Tₖ*uₖ - gradcache*Tₖ
        w = []
        W = []
        conscache .= zero(eltype(xₖ))
        cache.f.cons(conscache, xₖ)
        consjaccache .= zero(eltype(xₖ))
        cache.f.cons_j(consjaccache, xₖ)
        for j in 1:length(cache.lcons)
            if conscache[j] < 0
                push!(w, -α*conscache[j] - ϵ*min(dot(consjaccache[j,:], uₖ) + α*conscache[j], 0))
                push!(W, consjaccache[j,:])
            end
        end
        quadprob = OptimizationProblem(OptimizationFunction((v, p =nothing) -> (v - rₖ).^2, Optimization.AutoModelingToolkit(), cons = (res, v, p = nothing) -> res .= W .* v - w), uₖ, lcons = zeros(length(w)))
        
        uₖ = solve(quadprob, OSQP.Optimizer()).u
        xₖ = xₖ + Tₖ*uₖ
        if norm(uₖ, 1) < ϵ_sol
            break
        end
    end
    
    return SciMLBase.build_solution(cache, cache.opt, xₖ, opt.f(xₖ))
end

end