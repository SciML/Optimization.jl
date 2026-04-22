module OptimizationAuglag

using Reexport
using SciMLBase
@reexport using OptimizationBase
using SciMLBase: OptimizationProblem, OptimizationFunction, OptimizationStats
using LinearAlgebra: norm

export AugLag

struct AugLag{I, T, C, R}
    inner::I
    τ::T
    γ::T
    λmin::T
    λmax::T
    μmin::T
    μmax::T
    ϵ_primal::T
    ϵ_dual::T
    ρmax::T
    ρ_init::R
    progress_window::Int
    inner_maxiters::Union{Int, Nothing}
    inner_maxtime::Union{Float64, Nothing}
    inner_callback::C
end

function AugLag(;
        inner, τ = 0.5, γ = 10.0,
        λmin = -1.0e20, λmax = 1.0e20,
        μmin = 0.0, μmax = 1.0e20,
        ϵ = 1.0e-8, ϵ_primal = ϵ, ϵ_dual = ϵ,
        ρmax = 1.0e12, ρ_init = nothing, progress_window = 5,
        inner_maxiters = nothing,
        inner_maxtime = nothing,
        inner_callback = nothing
    )
    AugLag(
        inner, τ, γ, λmin, λmax, μmin, μmax,
        ϵ_primal, ϵ_dual, ρmax, ρ_init, progress_window,
        inner_maxiters, inner_maxtime, inner_callback
    )
end

SciMLBase.has_init(::AugLag) = true
SciMLBase.allowscallback(::AugLag) = true
SciMLBase.allowsbounds(::AugLag) = true
SciMLBase.requiresgradient(::AugLag) = true
SciMLBase.allowsconstraints(::AugLag) = true
SciMLBase.requiresconsjac(::AugLag) = true

function SciMLBase.__solve(cache::OptimizationCache{O}) where {O <: AugLag}
    maxiters = OptimizationBase._check_and_convert_maxiters(cache.solver_args.maxiters)
    maxtime = cache.solver_args.maxtime

    if isnothing(cache.f.cons)
        throw(ArgumentError("AugLag requires constraints; got cache.f.cons === nothing"))
    end

    eq_inds = [cache.lcons[i] == cache.ucons[i] for i in eachindex(cache.lcons)]
    ineq_inds = (!).(eq_inds)

    τ = cache.opt.τ
    γ = cache.opt.γ
    λmin = cache.opt.λmin
    λmax = cache.opt.λmax
    μmin = cache.opt.μmin
    μmax = cache.opt.μmax
    ϵ_primal = cache.opt.ϵ_primal
    ϵ_dual = cache.opt.ϵ_dual
    ρmax = cache.opt.ρmax
    W = cache.opt.progress_window
    inner_maxiters = cache.opt.inner_maxiters
    inner_maxtime = cache.opt.inner_maxtime
    inner_callback = cache.opt.inner_callback

    # kwargs forwarded to the inner solve — only include set ones so the
    # inner optimizer's own defaults apply otherwise.
    inner_kwargs = (;)
    isnothing(inner_maxiters) ||
        (inner_kwargs = merge(inner_kwargs, (; maxiters = inner_maxiters)))
    isnothing(inner_maxtime) ||
        (inner_kwargs = merge(inner_kwargs, (; maxtime = inner_maxtime)))
    isnothing(inner_callback) ||
        (inner_kwargs = merge(inner_kwargs, (; callback = inner_callback)))

    λ = zeros(eltype(cache.u0), sum(eq_inds))
    μ = zeros(eltype(cache.u0), sum(ineq_inds))

    cons_tmp = zeros(eltype(cache.u0), length(cache.lcons))
    cache.f.cons(cons_tmp, cache.u0)
    ρ = if isnothing(cache.opt.ρ_init)
        f_u0 = if OptimizationBase.isa_dataiterator(cache.p)
            cache.f(cache.u0, iterate(cache.p)[1])
        else
            cache.f(cache.u0, cache.p)
        end
        max(1.0e-6, min(10, 2 * abs(f_u0) / norm(cons_tmp)))
    else
        cache.opt.ρ_init
    end

    _loss = function (θ, p = cache.p)
        x = cache.f(θ, p)
        cons_tmp .= zero(eltype(θ))
        cache.f.cons(cons_tmp, θ)
        cons_tmp[eq_inds] .= cons_tmp[eq_inds] - cache.lcons[eq_inds]
        cons_tmp[ineq_inds] .= cons_tmp[ineq_inds] .- cache.ucons[ineq_inds]
        return x[1] + sum(@. λ * cons_tmp[eq_inds] + ρ / 2 * (cons_tmp[eq_inds] .^ 2)) +
               1 / (2 * ρ) * sum((max.(Ref(0.0), μ .+ (ρ .* cons_tmp[ineq_inds]))) .^ 2)
    end

    θ = copy(cache.u0)
    β = max.(cons_tmp[ineq_inds], -1 .* μ ./ ρ)
    eqidxs = [eq_inds[i] > 0 ? i : nothing for i in eachindex(ineq_inds)]
    ineqidxs = [ineq_inds[i] > 0 ? i : nothing for i in eachindex(ineq_inds)]
    eqidxs = eqidxs[eqidxs .!= nothing]
    ineqidxs = ineqidxs[ineqidxs .!= nothing]
    function aug_grad(G, θ, p)
        cache.f.grad(G, θ, p)
        if !isnothing(cache.f.cons_jac_prototype)
            J = similar(cache.f.cons_jac_prototype, eltype(θ))
        else
            J = zeros(eltype(θ), length(cache.lcons), length(θ))
        end
        cache.f.cons_j(J, θ)
        __tmp = zero(cons_tmp)
        cache.f.cons(__tmp, θ)
        __tmp[eq_inds] .= __tmp[eq_inds] .- cache.lcons[eq_inds]
        __tmp[ineq_inds] .= __tmp[ineq_inds] .- cache.ucons[ineq_inds]
        G .+= sum(
            λ[i] .* J[idx, :] + ρ * (__tmp[idx] .* J[idx, :])
                for (i, idx) in enumerate(eqidxs);
            init = zero(G)
        )
        return G .+= sum(
            1 / ρ * (max.(Ref(0.0), μ[i] .+ (ρ .* __tmp[idx])) .* J[idx, :])
                for (i, idx) in enumerate(ineqidxs);
            init = zero(G)
        )
    end

    opt_ret = ReturnCode.MaxIters

    augprob = OptimizationProblem(
        OptimizationFunction(_loss; grad = aug_grad), cache.u0, cache.p
    )

    t0 = time()
    if SciMLBase.has_init(cache.opt.inner)
        inner_cache = init(augprob, cache.opt.inner; inner_kwargs...)
    else
        inner_cache = augprob
    end

    n_outer = 0
    total_fevals = 0
    total_gevals = 0
    r_primal = Inf
    r_dual = Inf
    r_compl = Inf

    best_in_window = Inf
    best_prev_window = Inf
    iters_in_window = 0

    for i in 1:maxiters
        n_outer = i
        # warm-started inner solve of the augmented Lagrangian subproblem
        if SciMLBase.has_init(cache.opt.inner)
            SciMLBase.reinit!(inner_cache; u0 = θ)
            res = solve!(inner_cache)
        else
            new_prob = remake(inner_cache, u0 = θ)
            res = solve(new_prob, cache.opt.inner; maxiters = inner_maxiters)
        end
        total_fevals += res.stats.fevals
        total_gevals += res.stats.gevals

        θ = res.u
        cons_tmp .= 0.0
        cache.f.cons(cons_tmp, θ)

        λ = max.(min.(λmax, λ .+ ρ * (cons_tmp[eq_inds] .- cache.lcons[eq_inds])), λmin)
        β = max.(cons_tmp[ineq_inds], -1 .* μ ./ ρ)
        μ = min.(μmax, max.(μ .+ ρ * cons_tmp[ineq_inds], μmin))

        r_primal = norm(cons_tmp[eq_inds] .- cache.lcons[eq_inds], Inf)
        r_dual = ρ * r_primal
        r_compl = norm(β, Inf)

        # effective dual tolerance scales with ρ since at high ρ values
        # ||Δλ|| = ρ * ||c|| cannot go below `ϵ_primal * ρ` once primal
        # feasibility is achieved to the primal tolerance.
        ϵ_dual_eff = max(ϵ_dual, ϵ_primal * ρ)

        # outer-loop callback: one firing per outer iteration; AugLag-internal
        # state (residuals, multipliers, penalty) is exposed via `original`
        # for callers that want it.
        opt_state = OptimizationBase.OptimizationState(
            iter = i, u = θ, objective = res.objective,
            original = (; r_primal, r_dual, r_compl, λ, μ, ρ)
        )
        if cache.callback(opt_state, res.objective)
            opt_ret = ReturnCode.Terminated
            break
        end

        # convergence check
        if r_primal < ϵ_primal && r_dual < ϵ_dual_eff && r_compl < ϵ_primal
            opt_ret = ReturnCode.Success
            break
        end

        # instability detection
        if !isfinite(r_primal) || !isfinite(r_dual)
            opt_ret = ReturnCode.Unstable
            break
        end

        # maxtime enforced at outer-loop granularity
        if !isnothing(maxtime) && (time() - t0) > maxtime
            opt_ret = ReturnCode.MaxTime
            break
        end

        # window-based progress check compatible with stochastic optimizers
        worst = max(r_primal, r_compl)
        best_in_window = min(best_in_window, worst)
        iters_in_window += 1

        if iters_in_window ≥ W
            if best_in_window > τ * best_prev_window && best_in_window > ϵ_primal && ρ < ρmax
                ρ = min(ρmax, γ * ρ)
            end
            best_prev_window = best_in_window
            best_in_window = Inf
            iters_in_window = 0
        end
    end

    if opt_ret === ReturnCode.MaxIters && (r_primal ≥ ϵ_primal || r_compl ≥ ϵ_primal)
        opt_ret = ReturnCode.ConvergenceFailure
    end

    stats = OptimizationStats(;
        iterations = n_outer,
        time = time() - t0,
        fevals = total_fevals,
        gevals = total_gevals
    )

    obj = if OptimizationBase.isa_dataiterator(cache.p)
        total = 0.0
        n = 0
        for batch in cache.p
            total += cache.f(θ, batch)
            n += 1
        end
        total / n
    else
        cache.f(θ, cache.p)
    end

    return SciMLBase.build_solution(
        cache, cache.opt, θ, obj,
        stats = stats, retcode = opt_ret
    )
end

end
