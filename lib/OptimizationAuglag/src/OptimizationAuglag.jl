module OptimizationAuglag

using Reexport
using SciMLBase
@reexport using OptimizationBase
using SciMLBase: OptimizationProblem, OptimizationFunction, OptimizationStats
using LinearAlgebra: norm

export AugLag

include("auglag_function.jl")

struct AugLag{I, T, R, K}
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
    inner_kwargs::K
end

function AugLag(;
        inner, τ = 0.5, γ = 10.0,
        λmin = -1.0e20, λmax = 1.0e20,
        μmin = 0.0, μmax = 1.0e20,
        ϵ = 1.0e-8, ϵ_primal = ϵ, ϵ_dual = ϵ,
        ρmax = 1.0e12, ρ_init = nothing, progress_window = 5,
        inner_kwargs = (;)
    )
    return AugLag(
        inner, τ, γ, λmin, λmax, μmin, μmax,
        ϵ_primal, ϵ_dual, ρmax, ρ_init, progress_window,
        inner_kwargs
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
    inner_kwargs = cache.opt.inner_kwargs

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
        v0 = initial_violation(
            cons_tmp, cache.lcons, cache.ucons,
            eq_inds, ineq_upper_inds, ineq_lower_inds
        )
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
        OptimizationProblem(
            auglag_optf, cache.u0, cache.p;
            lb = cache.lb, ub = cache.ub
        )
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

        # Warm-started inner solve of the augmented Lagrangian subproblem.
        if SciMLBase.has_init(cache.opt.inner)
            SciMLBase.reinit!(inner_cache; u0 = θ)
            res = solve!(inner_cache)
        else
            new_prob = remake(inner_cache, u0 = θ)
            res = solve(new_prob, cache.opt.inner; inner_kwargs...)
        end
        total_fevals += res.stats.fevals
        total_gevals += res.stats.gevals

        θ = res.u
        cons_tmp .= zero(T)
        cache.f.cons(cons_tmp, θ, cache.p)
        ρ = ρ_ref[]

        # Multiplier updates. Equality: standard λ ← λ + ρ(c - l), clipped.
        # Inequalities: separate dual ascent on each finite-bounded side, with
        # `μ ≥ 0` enforced by the lower clamp (μmin defaults to 0).
        @inbounds for (i_loc, idx) in enumerate(eq_inds)
            λ[i_loc] = clamp(
                λ[i_loc] + ρ * (cons_tmp[idx] - cache.lcons[idx]),
                λmin, λmax
            )
        end
        @inbounds for (i_loc, idx) in enumerate(ineq_upper_inds)
            μ_upper[i_loc] = clamp(
                μ_upper[i_loc] + ρ * (cons_tmp[idx] - cache.ucons[idx]),
                μmin, μmax
            )
        end
        @inbounds for (i_loc, idx) in enumerate(ineq_lower_inds)
            μ_lower[i_loc] = clamp(
                μ_lower[i_loc] + ρ * (cache.lcons[idx] - cons_tmp[idx]),
                μmin, μmax
            )
        end

        r_primal, r_compl = primal_and_complementarity(
            cons_tmp, cache.lcons, cache.ucons,
            eq_inds, ineq_upper_inds, ineq_lower_inds,
            μ_upper, μ_lower, ρ
        )
        # ‖Δλ‖ scales with ρ‖c‖, so an effective dual tolerance bounded below
        # by ρ·ϵ_primal keeps convergence checks well-defined at large ρ.
        r_dual = ρ * r_primal
        ϵ_dual_eff = max(ϵ_dual, ϵ_primal * ρ)

        opt_state = OptimizationBase.OptimizationState(
            iter = i, u = θ, objective = res.objective,
            original = (;
                r_primal, r_dual, r_compl,
                λ, μ_upper, μ_lower, ρ,
            )
        )
        if cache.callback(opt_state, res.objective)
            opt_ret = ReturnCode.Terminated
            break
        end

        if r_primal < ϵ_primal && r_dual < ϵ_dual_eff && r_compl < ϵ_primal
            opt_ret = ReturnCode.Success
            break
        end

        if !isfinite(r_primal) || !isfinite(r_dual)
            opt_ret = ReturnCode.Unstable
            break
        end

        if !isnothing(maxtime) && (time() - t0) > maxtime
            opt_ret = ReturnCode.MaxTime
            break
        end

        # Window-based progress check, compatible with stochastic inner solvers.
        worst = max(r_primal, r_compl)
        best_in_window = min(best_in_window, worst)
        iters_in_window += 1

        if iters_in_window ≥ W
            if best_in_window > τ * best_prev_window &&
                    best_in_window > ϵ_primal && ρ < ρmax
                ρ_ref[] = min(ρmax, γ * ρ)
            end
            best_prev_window = best_in_window
            best_in_window = T(Inf)
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

function initial_violation(
        cons_tmp, lcons, ucons,
        eq_inds, ineq_upper_inds, ineq_lower_inds
    )
    v = zero(eltype(cons_tmp))
    @inbounds for idx in eq_inds
        v = max(v, abs(cons_tmp[idx] - lcons[idx]))
    end
    @inbounds for idx in ineq_upper_inds
        v = max(v, max(zero(v), cons_tmp[idx] - ucons[idx]))
    end
    @inbounds for idx in ineq_lower_inds
        v = max(v, max(zero(v), lcons[idx] - cons_tmp[idx]))
    end
    return v
end

function primal_and_complementarity(
        cons_tmp, lcons, ucons,
        eq_inds, ineq_upper_inds, ineq_lower_inds,
        μ_upper, μ_lower, ρ
    )
    T = eltype(cons_tmp)
    r_primal = zero(T)
    r_compl = zero(T)
    @inbounds for idx in eq_inds
        r_primal = max(r_primal, abs(cons_tmp[idx] - lcons[idx]))
    end
    @inbounds for (i_loc, idx) in enumerate(ineq_upper_inds)
        gu = cons_tmp[idx] - ucons[idx]
        r_primal = max(r_primal, max(zero(T), gu))
        β = max(gu, -μ_upper[i_loc] / ρ)
        r_compl = max(r_compl, abs(β))
    end
    @inbounds for (i_loc, idx) in enumerate(ineq_lower_inds)
        gl = lcons[idx] - cons_tmp[idx]
        r_primal = max(r_primal, max(zero(T), gl))
        β = max(gl, -μ_lower[i_loc] / ρ)
        r_compl = max(r_compl, abs(β))
    end
    return r_primal, r_compl
end

end
