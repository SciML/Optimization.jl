
@kwdef struct BFGS
    ϵ::Float64=1e-6
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
    g₀ = zeros(length(θ))
    f = cache.f
    p = cache.p
    ls = HagerZhang()
    _f = (θ) -> first(f.f(θ, p))

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

    Hₖ⁻¹= zeros(length(θ), length(θ))
    f.hess(Hₖ⁻¹, θ)
    Hₖ⁻¹ = inv(I(length(θ)) .+ Hₖ⁻¹)
    G = zeros(length(θ))
    f.grad(G, θ)
    s = -1 * Hₖ⁻¹ * G

    t0 = time()
    for i in 1:maxiters
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
                _ϕdϕ = ϕdϕ(θ, dir)
        
                ϕ₀, dϕ₀ = _ϕdϕ(zero(eltype(θ)))
                ls(_ϕ, _ϕdϕ, 1.0, ϕ₀, dϕ₀)[1]
            catch err
                1.0
            end
        end

        θ = θ .+ αₖ.*pₖ
        s = αₖ.*pₖ

        G = zeros(length(θ))
        f.grad(G, θ)
        zₖ = G - q
        ρₖ = 1/dot(zₖ, s)
        Hₖ⁻¹ = mul!((I - ρₖ*s*zₖ'), Hₖ⁻¹, (I - ρₖ*zₖ*s')) + ρₖ*(s*s')
        if norm(G, Inf) <= opt.ϵ
            println(i)
            break
        end
    end

    t1 = time()

    SciMLBase.build_solution(cache, cache.opt, θ, f(θ, cache.p), solve_time = t1 - t0)
    # here should be build_solution to create the output message
end
