module OptimizationAuglag

using Reexport
using SciMLBase
@reexport using OptimizationBase
using SciMLBase: OptimizationProblem, OptimizationFunction, OptimizationStats
using LinearAlgebra: norm

export AugLag

include("auglag_function.jl")

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
SciMLBase.allowsfg(::AugLag) = true
SciMLBase.allowsconstraints(::AugLag) = true
SciMLBase.requiresconsjac(::AugLag) = true

function SciMLBase.__solve(cache::OptimizationCache{O}) where {O <: AugLag}
    maxiters = OptimizationBase._check_and_convert_maxiters(cache.solver_args.maxiters)
    maxtime = cache.solver_args.maxtime

    if isnothing(cache.f.cons)
        throw(ArgumentError("AugLag requires constraints; got cache.f.cons === nothing"))
    end

    eq_inds, ineq_upper_inds, ineq_lower_inds = classify_constraints(
        cache.lcons, cache.ucons
    )

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

    inner_kwargs = (;)
    isnothing(inner_maxiters) ||
        (inner_kwargs = merge(inner_kwargs, (; maxiters = inner_maxiters)))
    isnothing(inner_maxtime) ||
        (inner_kwargs = merge(inner_kwargs, (; maxtime = inner_maxtime)))
    isnothing(inner_callback) ||
        (inner_kwargs = merge(inner_kwargs, (; callback = inner_callback)))

    T = eltype(cache.u0)
    λ = zeros(T, length(eq_inds))
    μ_upper = zeros(T, length(ineq_upper_inds))
    μ_lower = zeros(T, length(ineq_lower_inds))
    ρ_ref = Ref(zero(T))

    cons_tmp = zeros(T, length(cache.lcons))
    cache.f.cons(cons_tmp, cache.u0, cache.p)

    # Initial penalty: scale to (very roughly) balance objective and worst
    # constraint violation at u0. Falls back to 1.0 when u0 is already feasible
    # (no violations to scale against) or when the user pins ρ_init.
    ρ_ref[] = if isnothing(cache.opt.ρ_init)
        f_u0 = if OptimizationBase.isa_dataiterator(cache.p)
            cache.f(cache.u0, iterate(cache.p)[1])
        else
            cache.f(cache.u0, cache.p)
        end
        v0 = initial_violation(cons_tmp, cache.lcons, cache.ucons,
            eq_inds, ineq_upper_inds, ineq_lower_inds)
        v0 < eps(T) ? one(T) :
        T(max(1.0e-6, min(10.0, 2 * abs(first(f_u0)) / v0)))
    else
        T(cache.opt.ρ_init)
    end

    auglag_optf = generate_auglag(
        cache, eq_inds, ineq_upper_inds, ineq_lower_inds,
        λ, μ_upper, μ_lower, ρ_ref
    )

    # Forward θ-bounds only when the inner solver can use them; otherwise the
    # bounds are honored only as far as AugLag itself constrains things.
    augprob = if cache.lb !== nothing && cache.ub !== nothing &&
            SciMLBase.allowsbounds(cache.opt.inner)
        OptimizationProblem(auglag_optf, cache.u0, cache.p;
            lb = cache.lb, ub = cache.ub)
    else
        OptimizationProblem(auglag_optf, cache.u0, cache.p)
    end

    t0 = time()
    if SciMLBase.has_init(cache.opt.inner)
        inner_cache = init(augprob, cache.opt.inner; inner_kwargs...)
    else
        inner_cache = augprob
    end

    θ = copy(cache.u0)
    opt_ret = ReturnCode.MaxIters

    n_outer = 0
    total_fevals = 0
    total_gevals = 0
    r_primal = T(Inf)
    r_dual = T(Inf)
    r_compl = T(Inf)

    best_in_window = T(Inf)
    best_prev_window = T(Inf)
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
