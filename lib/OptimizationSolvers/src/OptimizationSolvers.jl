module OptimizationSolvers

using Reexport, Printf, ProgressLogging
@reexport using Optimization
using Optimization.SciMLBase, LineSearches

struct BFGS
    ϵ::Float64
    m::Int
end

SciMLBase.supports_opt_cache_interface(opt::BFGS) = true
include("sophia.jl")

function SciMLBase.__init(prob::SciMLBase.OptimizationProblem, opt::BFGS,
    data = Optimization.DEFAULT_DATA; save_best = true,
    callback = (args...) -> (false),
    progress = false, kwargs...)
    return OptimizationCache(prob, opt, data; save_best, callback, progress,
        kwargs...)
end


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
    O <:BFGS,
    D,
    P,
    C,
}
    if cache.data != Optimization.DEFAULT_DATA
        maxiters = length(cache.data)
        data = cache.data
    else
        maxiters = Optimization._check_and_convert_maxiters(cache.solver_args.maxiters)
        data = Optimization.take(cache.data, maxiters)
    end
    opt = cache.opt
    θ = copy(cache.u0)
    G = zeros(length(θ))
    f = cache.f

    _f = (θ) -> first(f.f(θ, cache.p))

    ϕ(α) = _f(θ .+ α.*s)
    function dϕ(α)
        f.grad(G, θ .+ α.*s)
        return dot(G, s)
    end
    function ϕdϕ(α)
        phi = _f(θ .+ α.*s)
        f.grad(G, θ .+ α.*s)
        dphi = dot(G, s)
        return (phi, dphi)
    end
    Hₖ⁻¹= zeros(length(θ), length(θ))
    f.hess(Hₖ⁻¹, θ)
    println(Hₖ⁻¹)
    Hₖ⁻¹ = inv(I(length(θ)) .+ Hₖ⁻¹)
    f.grad(G, θ)
    s = -1 * Hₖ⁻¹ * G
    # m = opt.m
    # α = Vector{typeof(θ)}(undef, m)
    # β = zeros(m)
    # ss = Vector{typeof(θ)}(undef, m)
    # y = Vector{typeof(θ)}(undef, m)
    # ρ = Vector{Float64}(undef, m)
    # ρ[1] = 1.0
    # k = 1
    # t0 = time()
    # ss[1] = θ
    # f.grad(G, θ)
    # y[1] = G
    # α[1] = α0
    # γ = dot(ss[1], y[1])/dot(y[1], y[1])
    # Hₖ = I(length(θ)) * γ
    # ρ[1] = 1/dot(y[1], ss[1])
    
    t0 = time()
    for i in 1:maxiters
        println(i, " ", θ, " Objective: ", f(θ, cache.p))
        # println(ss, " ", y, " ", γ)
        
        q = copy(G)
        # if k > 1
        #     y[k-1] = q - y[k-1]
        #     γ = dot(ss[k-1], y[k-1])/dot(y[k-1], y[k-1])
        #     Hₖ = I(length(θ)) * γ

        #     ρ[k] = 1/dot(y[k-1], ss[k-1])
        # end
        
        # for j in 1:min(m,i-1)
        #     α[j] = ρ[j]*dot(ss[j], G)
        #     G = G - α[j]*y[j]
        # end
        # r = Hₖ*G
        # for j in min(m,i-1):1
        #     β[j] = ρ[j]*dot(y[j], r)
        #     r = r + ss[j]*(α[j] - β[j])
        # end
        pₖ = -Hₖ⁻¹* G
        fx = _f(θ)
        dir = dot(G, pₖ)
        println(fx, " ", dir)

        if dir > 0
            pₖ = -G
            dir = dot(G, pₖ)
        end

        αₖ = let 
            try
                [(HagerZhang())(ϕ, dϕ, ϕdϕ, 1.0, fx, dir)...]
            catch err
                αₖ = [1.0]
            end
        end
        # α[k] = αₖ
        
        θ = θ .+ αₖ.*pₖ
        s = αₖ.*pₖ

        # if k > m
        #     ss[1:end-1] = ss[2:end]
        #     y[1:end-1] = y[2:end]
        #     k = m
        #     ss[k] =  α[k-2]*pₖ
        #     y[k] = q
        #     α[1:end-1] = α[2:end]
        # end
        # k+=1
        G = zeros(length(θ))
        f.grad(G, θ)
        zₖ = G - q
        Hₖ⁻¹ = (I - (s*zₖ')/dot(zₖ, s))*Hₖ⁻¹*(I - (zₖ*s')/dot(zₖ, s)) + (s*s')/dot(zₖ, s)
        if norm(G) < 1e-6
            break
        end
    end


    t1 = time()

    SciMLBase.build_solution(cache, cache.opt, θ, f(θ, cache.p), solve_time = t1 - t0)
    # here should be build_solution to create the output message
end

end
