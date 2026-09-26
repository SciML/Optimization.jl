"""
    classify_constraints(lcons, ucons)

Partition constraint indices given the SciMLBase contract
`lcons[i] ≤ c_i(θ) ≤ ucons[i]`:

  - `eq_inds`: rows where `lcons[i] == ucons[i]` (equality).
  - `ineq_upper_inds`: rows with `lcons[i] != ucons[i]` and finite `ucons[i]`,
    contributing the penalty for `c_i ≤ ucons[i]`.
  - `ineq_lower_inds`: rows with `lcons[i] != ucons[i]` and finite `lcons[i]`,
    contributing the penalty for `lcons[i] ≤ c_i`.

A two-sided inequality (both bounds finite, `lcons[i] < ucons[i]`) appears in
*both* `ineq_upper_inds` and `ineq_lower_inds` and gets a separate Lagrange
multiplier per side.
"""
function classify_constraints(lcons, ucons)
    eq_inds = Int[]
    ineq_upper_inds = Int[]
    ineq_lower_inds = Int[]
    @inbounds for i in eachindex(lcons)
        l, u = lcons[i], ucons[i]
        if l == u
            push!(eq_inds, i)
        else
            isfinite(u) && push!(ineq_upper_inds, i)
            isfinite(l) && push!(ineq_lower_inds, i)
        end
    end
    return eq_inds, ineq_upper_inds, ineq_lower_inds
end

"""
    generate_auglag(cache, eq_inds, ineq_upper_inds, ineq_lower_inds,
                    λ, μ_upper, μ_lower, ρ_ref)

Build the augmented-Lagrangian subproblem function as an `OptimizationFunction`
with analytical value, gradient, and `fg!` derived from the user's loss and
constraints in `cache.f`.

The augmented Lagrangian is

    L(θ; λ, μ_u, μ_l, ρ) = f(θ)
        + Σᵢ∈eq    [ λᵢ (cᵢ - lᵢ) + (ρ/2)(cᵢ - lᵢ)² ]
        + (1/(2ρ)) Σᵢ∈up max(0, μ_uᵢ + ρ (cᵢ - uᵢ))²
        + (1/(2ρ)) Σᵢ∈lo max(0, μ_lᵢ + ρ (lᵢ - cᵢ))²

so its gradient (used closed-form, not by AD'ing through `L`) is

    ∇L = ∇f
        + Σᵢ∈eq    (λᵢ + ρ (cᵢ - lᵢ)) ∇cᵢ
        + Σᵢ∈up    max(0, μ_uᵢ + ρ (cᵢ - uᵢ)) ∇cᵢ
        - Σᵢ∈lo    max(0, μ_lᵢ + ρ (lᵢ - cᵢ)) ∇cᵢ.

`λ`, `μ_upper`, `μ_lower` are mutable vectors and `ρ_ref` is a `Ref{<:Real}`
(or any zero-arg-getindex container). The closures dereference them at call
time, so the outer AugLag loop can update multipliers and the penalty in
place between inner solves without rebuilding the function.

The constraint term of `∇L` is a single vector-Jacobian product `Jᵀv` with
`cache.f.cons_vjp`, where `v` holds the multiplier weight of every constraint
row (zero for inactive inequalities). `AugLag` declares `requiresconsvjp`, so
OptimizationBase always provides one: the user's, one built from the user's
`cons_j`, the AD backend's pullback, or a chunked/colored AD Jacobian for
forward-mode backends. Because inactive rows enter with a zero weight rather
than being skipped, a non-finite derivative in an inactive row propagates to
the gradient.

`cons_tmp`, `v` and `Jᵀv` are preallocated once with element type
`eltype(cache.u0)`. This is safe because the analytical gradient does not AD
through this function — `cache.f.grad` and `cache.f.cons_vjp` handle their own
AD internally.

# Constraints and the data-iterator `p`

Constraints are treated as **non-stochastic / batch-independent**: the
user's `cons!(res, θ, p)` is always invoked with the *full* `cache.p`,
regardless of which inner-solve batch is currently being processed. For
the typical non-`DataLoader` case `cache.p` is just the user's `p`. For a
`DataLoader` `p`, `cons!` receives the iterator itself, and the user's
constraint body may pull the underlying full data from it (e.g. via
`p.data` for an `MLUtils.DataLoader`) — but the constraint is still a
deterministic function of `θ` and the full data, never of a single batch.

Consistent with this, `cache.f.cons_vjp` is invoked without `p` and uses
the `p` that was closed over at instantiation (the first batch for a data
iterator). Since the constraint is by contract batch-independent, that
closed-over `p` is irrelevant to its value.
"""
function generate_auglag(
        cache,
        eq_inds, ineq_upper_inds, ineq_lower_inds,
        λ, μ_upper, μ_lower, ρ_ref
    )
    n = length(cache.u0)
    m = length(cache.lcons)
    T = eltype(cache.u0)

    cons_tmp = zeros(T, m)
    Jᵀv = zeros(T, n)
    v = zeros(T, m)

    lcons = cache.lcons
    ucons = cache.ucons

    multipliers! = function (v, cons_tmp, ρ, f_val)
        fill!(v, zero(T))
        L = f_val

        @inbounds for (i, idx) in enumerate(eq_inds)
            ce = cons_tmp[idx] - lcons[idx]
            v[idx] += λ[i] + ρ * ce
            L += λ[i] * ce + (ρ / 2) * ce^2
        end
        @inbounds for (i, idx) in enumerate(ineq_upper_inds)
            cu = cons_tmp[idx] - ucons[idx]
            m_act = max(zero(T), μ_upper[i] + ρ * cu)
            v[idx] += m_act
            L += m_act^2 / (2 * ρ)
        end
        @inbounds for (i, idx) in enumerate(ineq_lower_inds)
            cl = lcons[idx] - cons_tmp[idx]
            m_act = max(zero(T), μ_lower[i] + ρ * cl)
            v[idx] -= m_act
            L += m_act^2 / (2 * ρ)
        end

        return L
    end

    auglag_value = function (θ, p)
        f_val = first(cache.f(θ, p))
        cache.f.cons(cons_tmp, θ, cache.p)
        ρ = ρ_ref[]
        L = multipliers!(v, cons_tmp, ρ, f_val)
        return L
    end

    auglag_grad! = function (G, θ, p)
        cache.f.grad(G, θ, p)
        cache.f.cons(cons_tmp, θ, cache.p)

        ρ = ρ_ref[]
        multipliers!(v, cons_tmp, ρ, zero(T))
        cache.f.cons_vjp(Jᵀv, θ, v)

        G .+= Jᵀv

        return G
    end

    auglag_fg! = function (G, θ, p)
        f_val = if !isnothing(cache.f.fg)
            first(cache.f.fg(G, θ, p))
        else
            cache.f.grad(G, θ, p)
            first(cache.f(θ, p))
        end
        cache.f.cons(cons_tmp, θ, cache.p)

        ρ = ρ_ref[]
        L = multipliers!(v, cons_tmp, ρ, f_val)
        cache.f.cons_vjp(Jᵀv, θ, v)

        G .+= Jᵀv

        return L
    end

    return OptimizationFunction(
        auglag_value; grad = auglag_grad!, fg = auglag_fg!
    )
end
