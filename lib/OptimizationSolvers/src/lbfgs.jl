struct LBFGS
    ϵ::Float64
    m::Int
end

SciMLBase.supports_opt_cache_interface(opt::LBFGS) = true

function SciMLBase.__init(prob::SciMLBase.OptimizationProblem, opt::LBFGS,
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
    O <:LBFGS,
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

    Sₖ = zeros(length(θ), opt.m)
    Yₖ = zeros(length(θ), opt.m)
    Rₖ = zeros(opt.m, opt.m)
    Dₖ = zeros(opt.m)

    Hₖ⁻¹= zeros(length(θ), length(θ))
    println(Hₖ⁻¹)
    Hₖ⁻¹ = I(length(θ))
    f.grad(G, θ)
    s = -1 * Hₖ⁻¹ * G
    t0 = time()
    for k in 1:opt.m
        # println(k, " ", θ, " Objective: ", f(θ, cache.p))
        q = copy(G)
        pₖ = -Hₖ⁻¹* G
        fx = _f(θ)
        dir = dot(G, pₖ)
        # println(fx, " ", dir)
        if !isnan(dir) && dir > 0
            pₖ = -G
            dir = dot(G, pₖ)
        else
            dir = -G
        end
        αₖ = let
            try
                [(HagerZhang())(ϕ, dϕ, ϕdϕ, 1.0, fx, dir)...]
            catch err
                αₖ = [1.0]
            end
        end
        θ = θ .+ αₖ.*pₖ
        s = αₖ.*pₖ
        G = zeros(length(θ))
        f.grad(G, θ)
        zₖ = G - q
        Hₖ⁻¹ = (I - (s*zₖ')/dot(zₖ, s))*Hₖ⁻¹*(I - (zₖ*s')/dot(zₖ, s)) + (s*s')/dot(zₖ, s)
        Sₖ[:, k] .= s
        Yₖ[:, k] .= zₖ
        if norm(G) < 1e-6
            break
        end
    end

    for j in 1:opt.m
        for i in 1:j
            Rₖ[i, j] = dot(Sₖ[:, i], Yₖ[:, j])
        end
        Dₖ[j] = dot(Sₖ[:, j], Yₖ[:, j])
    end

    m = opt.m
    for i in opt.m+1:maxiters
        _G = copy(G)
        fx = _f(θ)
        println(i, " ", θ, " Objective: ", fx)
        γₖ = dot(Sₖ[:, m], Yₖ[:, m])/dot(Yₖ[:, m], Yₖ[:, m])
        Rinv = let
            try
                inv(Rₖ)
            catch
                println(i, " ", Rₖ)
                println("Inversion failed")
                break
            end
        end

        p = [Rinv'*(diagm(Dₖ) + γₖ * Yₖ'*Yₖ)*Rinv*(Sₖ'*G) - γₖ * Rinv*(Yₖ'*G); -Rinv*(Sₖ'*G)]
        p = -1 .* (γₖ * G + hcat(Sₖ, γₖ*Yₖ)*p)
        p = dot(p, G) 
        αₖ = let
            try
                [(HagerZhang())(ϕ, dϕ, ϕdϕ, 1.0, fx, p)...]
            catch err
                println(err)
                break
                αₖ = [1.0]
            end
        end

        θ = θ .+ αₖ.*p
        s = αₖ.*p
        Sₖ[:, 1:end-1] .= Sₖ[:, 2:end]
        Yₖ[:, 1:end-1] .= Yₖ[:, 2:end]
        Dₖ[1:end-1] .= Dₖ[2:end]
        Rₖ[1:end-1, 1:end-1] .= Rₖ[2:end, 2:end]
        Sₖ[:, end] .= s
        G = zeros(length(θ))
        f.grad(G, θ)
        zₖ = G - _G
        Yₖ[:, end] .= zₖ
        for i in 1:m
            Rₖ[i, m] = dot(Sₖ[:, i], Yₖ[:, m])
        end
        Dₖ[m] = dot(Sₖ[:, m], Yₖ[:, m])
    end

    t1 = time()

    SciMLBase.build_solution(cache, cache.opt, θ, f(θ, cache.p), solve_time = t1 - t0)
    # here should be build_solution to create the output message
end