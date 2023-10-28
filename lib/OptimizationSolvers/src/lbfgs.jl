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
    g₀ = zeros(length(θ))
    f = cache.f

    _f = (θ) -> first(f.f(θ, cache.p))

    function ϕ(u, du)
        function ϕ_internal(α)
            u_ = u - α * du
            _fu = _f(u_)
            return dot(_fu, _fu) / 2
        end
        return ϕ_internal
    end

    function dϕ(u, du)
        function dϕ_internal(α)
            u_ = u - α * du
            _fu = _f(u_)
            f.grad(g₀, u_)
            return dot(g₀, -du)
        end
        return dϕ_internal
    end

    function ϕdϕ(u, du)
        function ϕdϕ_internal(α)
            u_ = u - α * du
            _fu = _f(u_)
            f.grad(g₀, u_)
            return dot(_fu, _fu) / 2, dot(g₀, -du)
        end
        return ϕdϕ_internal
    end

    Sₖ = zeros(length(θ), opt.m)
    Yₖ = zeros(length(θ), opt.m)
    Rₖ = zeros(opt.m, opt.m)
    Dₖ = zeros(opt.m)

    Hₖ⁻¹= zeros(length(θ), length(θ))
    Hₖ⁻¹ = I(length(θ))
    G = zeros(length(θ))
    f.grad(G, θ)
    s = -1 * Hₖ⁻¹ * G
    t0 = time()
    conv = false
    for k in 1:opt.m
        q = copy(G)
        pₖ = -Hₖ⁻¹* G
        fx = _f(θ)
        dir = -pₖ
        if all(isnan.(dir)) || all(dir .> 0)
            pₖ = -G
            dir = G
        end
        αₖ = let
            try
                _ϕ = ϕ(θ, dir)
                _dϕ = dϕ(θ, dir)
                _ϕdϕ = ϕdϕ(θ, dir)
        
                ϕ₀, dϕ₀ = _ϕdϕ(zero(eltype(θ)))
                (HagerZhang())(_ϕ, _dϕ, _ϕdϕ, 1.0, ϕ₀, dϕ₀)[1]
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
        if norm(G) < opt.ϵ
            conv = true
            break
        end
    end

    if !conv
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
            γₖ = dot(Sₖ[:, m], Yₖ[:, m])/dot(Yₖ[:, m], Yₖ[:, m])
            Rinv = let
                try
                    inv(Rₖ)
                catch
                    println("Inversion failed")
                    break
                end
            end

            p = [Rinv'*(diagm(Dₖ) + γₖ * Yₖ'*Yₖ)*Rinv*(Sₖ'*G) - γₖ * Rinv*(Yₖ'*G); -Rinv*(Sₖ'*G)]
            p = -1 .* (γₖ * G + hcat(Sₖ, γₖ*Yₖ)*p)
            dir = -p 
            αₖ = let
                try
                    _ϕ = ϕ(θ, dir)
                    _dϕ = dϕ(θ, dir)
                    _ϕdϕ = ϕdϕ(θ, dir)
                
                    ϕ₀, dϕ₀ = _ϕdϕ(zero(eltype(θ)))
                    (HagerZhang())(_ϕ, _dϕ, _ϕdϕ, 1.0, ϕ₀, dϕ₀)[1]
                catch err
                    αₖ = 1.0
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
            if norm(G) < opt.ϵ
                break
            end
        end
    end
    t1 = time()

    SciMLBase.build_solution(cache, cache.opt, θ, f(θ, cache.p), solve_time = t1 - t0)
    # here should be build_solution to create the output message
end