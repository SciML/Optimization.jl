function generate_auglag(θ)
    x = cache.f(θ, cache.p)
    cons_tmp .= zero(eltype(θ))
    cache.f.cons(cons_tmp, θ)
    cons_tmp[eq_inds] .= cons_tmp[eq_inds] - cache.lcons[eq_inds]
    cons_tmp[ineq_inds] .= cons_tmp[ineq_inds] .- cache.ucons[ineq_inds]
    opt_state = OptimizationBase.OptimizationState(u = θ, objective = x[1])
    if cache.callback(opt_state, x...)
        error("Optimization halted by callback.")
    end
    return x[1] + sum(@. λ * cons_tmp[eq_inds] + ρ / 2 * (cons_tmp[eq_inds] .^ 2)) +
           1 / (2 * ρ) * sum((max.(Ref(0.0), μ .+ (ρ .* cons_tmp[ineq_inds]))) .^ 2)
end
