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
    generate_auglag(
        cache, eq_inds, ineq_upper_inds, ineq_lower_inds,
        λ, μ_upper, μ_lower, ρ_ref
    )

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

`cons_tmp` and `J` are preallocated once with element type `eltype(cache.u0)`.
This is safe because the analytical gradient does not AD through this
function — `cache.f.grad` and `cache.f.cons_j` (which the inner solver
ultimately calls) handle their own AD internally.

# Constraints and the data-iterator `p`

Constraints are treated as **non-stochastic / batch-independent**: the
user's `cons!(res, θ, p)` is always invoked with the *full* `cache.p`,
regardless of which inner-solve batch is currently being processed. For
the typical non-`DataLoader` case `cache.p` is just the user's `p`. For a
`DataLoader` `p`, `cons!` receives the iterator itself, and the user's
constraint body may pull the underlying full data from it (e.g. via
`p.data` for an `MLUtils.DataLoader`) — but the constraint is still a
deterministic function of `θ` and the full data, never of a single batch.

Consistent with this, the constraint Jacobian `cache.f.cons_j` is invoked
without `p` and uses the `p` that was closed over at AD-preparation time
(the first batch for a data iterator). Since the constraint is by
contract batch-independent, that closed-over `p` is irrelevant to the
Jacobian's value.
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
    J = zeros(T, m, n)
    lcons = cache.lcons
    ucons = cache.ucons

    auglag_value = function (θ, p)
        f_val = first(cache.f(θ, p))
        cache.f.cons(cons_tmp, θ, cache.p)
        ρ = ρ_ref[]
        L = f_val
        @inbounds for (i, idx) in enumerate(eq_inds)
            ce = cons_tmp[idx] - lcons[idx]
            L += λ[i] * ce + (ρ / 2) * ce^2
        end
        @inbounds for (i, idx) in enumerate(ineq_upper_inds)
            cu = cons_tmp[idx] - ucons[idx]
            m_act = max(zero(T), μ_upper[i] + ρ * cu)
            L += m_act^2 / (2 * ρ)
        end
        @inbounds for (i, idx) in enumerate(ineq_lower_inds)
            cl = lcons[idx] - cons_tmp[idx]
            m_act = max(zero(T), μ_lower[i] + ρ * cl)
            L += m_act^2 / (2 * ρ)
        end
        return L
    end

    auglag_grad! = function (G, θ, p)
        cache.f.grad(G, θ, p)
        cache.f.cons(cons_tmp, θ, cache.p)
        cache.f.cons_j(J, θ)
        ρ = ρ_ref[]
        @inbounds for (i, idx) in enumerate(eq_inds)
            ce = cons_tmp[idx] - lcons[idx]
            a = λ[i] + ρ * ce
            @views @. G += a * J[idx, :]
        end
        @inbounds for (i, idx) in enumerate(ineq_upper_inds)
            cu = cons_tmp[idx] - ucons[idx]
            m_act = max(zero(T), μ_upper[i] + ρ * cu)
            if m_act > zero(T)
                @views @. G += m_act * J[idx, :]
            end
        end
        @inbounds for (i, idx) in enumerate(ineq_lower_inds)
            cl = lcons[idx] - cons_tmp[idx]
            m_act = max(zero(T), μ_lower[i] + ρ * cl)
            if m_act > zero(T)
                @views @. G -= m_act * J[idx, :]
            end
        end
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
        cache.f.cons_j(J, θ)
        ρ = ρ_ref[]
        L = f_val
        @inbounds for (i, idx) in enumerate(eq_inds)
            ce = cons_tmp[idx] - lcons[idx]
            a = λ[i] + ρ * ce
            L += λ[i] * ce + (ρ / 2) * ce^2
            @views @. G += a * J[idx, :]
        end
        @inbounds for (i, idx) in enumerate(ineq_upper_inds)
            cu = cons_tmp[idx] - ucons[idx]
            m_act = max(zero(T), μ_upper[i] + ρ * cu)
            if m_act > zero(T)
                L += m_act^2 / (2 * ρ)
                @views @. G += m_act * J[idx, :]
            end
        end
        @inbounds for (i, idx) in enumerate(ineq_lower_inds)
            cl = lcons[idx] - cons_tmp[idx]
            m_act = max(zero(T), μ_lower[i] + ρ * cl)
            if m_act > zero(T)
                L += m_act^2 / (2 * ρ)
                @views @. G -= m_act * J[idx, :]
            end
        end
        return L
    end

    return OptimizationFunction(
        auglag_value; grad = auglag_grad!, fg = auglag_fg!
    )
end
