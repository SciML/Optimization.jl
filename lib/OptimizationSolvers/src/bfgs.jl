
struct BFGS
    ϵ::Float64
    m::Int
end

SciMLBase.supports_opt_cache_interface(opt::BFGS) = true


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

    t0 = time()
    for i in 1:maxiters
        println(i, " ", θ, " Objective: ", f(θ, cache.p))
        q = copy(G)
        @show q
        pₖ = -Hₖ⁻¹* G
        fx = _f(θ)
        dir = G' * pₖ
        println(dir)

        if isnan(dir) || dir > 0
            pₖ = -G
            dir = -G'*G
        end

        αₖ = let
            try
                (HagerZhang())(ϕ, dϕ, ϕdϕ, 1.0, fx, dir)[1]
            catch err
                println(err)
                1.0
            end
        end

        θ = θ .+ αₖ.*pₖ
        s = αₖ.*pₖ

        G = zeros(length(θ))
        f.grad(G, θ)
        zₖ = G - q
        @show G
        ρₖ = 1/dot(zₖ, s)
        Hₖ⁻¹ = (I - ρₖ*s*zₖ')*Hₖ⁻¹*(I - ρₖ*zₖ*s') + ρₖ*(s*s')
        if norm(G) <= opt.ϵ
            break
        end
    end


    t1 = time()

    SciMLBase.build_solution(cache, cache.opt, θ, f(θ, cache.p), solve_time = t1 - t0)
    # here should be build_solution to create the output message
end
