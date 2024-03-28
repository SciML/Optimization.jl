using Optimization.SciMLBase, LBFGSB

"""
$(TYPEDEF)

[L-BFGS-B](https://en.wikipedia.org/wiki/Limited-memory_BFGS#L-BFGS-B) Nonlinear Optimization Code from [LBFGSB](https://github.com/Gnimuc/LBFGSB.jl/tree/master).
It is a quasi-Newton optimization algorithm that supports bounds.

References

- R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound Constrained Optimization, (1995), SIAM Journal on Scientific and Statistical Computing , 16, 5, pp. 1190-1208.
- C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B, FORTRAN routines for large scale bound constrained optimization (1997), ACM Transactions on Mathematical Software, Vol 23, Num. 4, pp. 550 - 560.
- J.L. Morales and J. Nocedal. L-BFGS-B: Remark on Algorithm 778: L-BFGS-B, FORTRAN routines for large scale bound constrained optimization (2011), to appear in ACM Transactions on Mathematical Software.
"""
@kwdef struct LBFGS
    m::Int = 10
end

SciMLBase.supports_opt_cache_interface(::LBFGS) = true
SciMLBase.allowsbounds(::LBFGS) = true
# SciMLBase.requiresgradient(::LBFGS) = true
SciMLBase.allowsconstraints(::LBFGS) = true

function SciMLBase.__init(prob::SciMLBase.OptimizationProblem,
        opt::LBFGS,
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
        C
}) where {
        F,
        RC,
        LB,
        UB,
        LC,
        UC,
        S,
        O <:
        LBFGS,
        D,
        P,
        C
}
    if cache.data != Optimization.DEFAULT_DATA
        maxiters = length(cache.data)
        data = cache.data
    else
        maxiters = Optimization._check_and_convert_maxiters(cache.solver_args.maxiters)
        data = Optimization.take(cache.data, maxiters)
    end

    local x

    if !isnothing(cache.f.cons) 
        eq_inds = [cache.lcons[i] == cache.ucons[i] for i in eachindex(cache.lcons)]
        ineq_inds = (!).(eq_inds)

        τ = 0.5
        γ = 10.0
        λmin = -1e20
        λmax = 1e20
        μmin = 0.0
        μmax = 1e20
        ϵ = 1e-8

        λ = zeros(eltype(cache.u0), sum(eq_inds))
        μ = zeros(eltype(cache.u0), sum(ineq_inds))

        cons_tmp = zeros(eltype(cache.u0), length(cache.lcons))
        cache.f.cons(cons_tmp, cache.u0)
        ρ = max(1e-6, min(10, 2*(abs(cache.f(cache.u0, cache.p)))/ norm(cons_tmp) ))

        _loss = function (θ)
            x = cache.f(θ, cache.p)
            cons_tmp .= 0.0 
            cache.f.cons(cons_tmp, θ)
            opt_state = Optimization.OptimizationState(u = θ, objective = x[1])
            if cache.callback(opt_state, x...)
                error("Optimization halted by callback.")
            end
            return x[1] + sum(@. λ * cons_tmp[eq_inds] + ρ/2 * (cons_tmp[eq_inds].^2)) + 0.5/ ρ * sum((max.(Ref(0.0), μ .+ (ρ .* cons_tmp[ineq_inds]))).^2)
        end

        prev_eqcons = zero(λ)
        θ = cache.u0
        β = max.(cons_tmp[ineq_inds], Ref(0.0))
        prevβ = zero(β)
        eqidxs =  [eq_inds[i] > 0 ? i : nothing for i in eachindex(ineq_inds)]
        ineqidxs = [ineq_inds[i] > 0 ? i : nothing for i in eachindex(ineq_inds)]
        eqidxs = eqidxs[eqidxs.!=nothing]
        ineqidxs = ineqidxs[ineqidxs.!=nothing]
        function aug_grad(G, θ)
            cache.f.grad(G, θ)
            J = zeros((length(cache.lcons), length(θ)))
            cache.f.cons_j(J, θ)
            __tmp = zero(cons_tmp)
            cache.f.cons(__tmp, θ)
            G .+= sum(λ .* J[i, :] + ρ * (__tmp[eq_inds].* J[i, :]) for i in eqidxs)
            G .+= sum(ρ * (max.(Ref(0.0), μ .+ (ρ .* __tmp[ineq_inds])) .* J[i, :]) for i in ineqidxs)
        end
        for i in 1:maxiters
            prev_eqcons .= cons_tmp[eq_inds]
            prevβ .= copy(β)
            if cache.lb !== nothing && cache.ub !== nothing
                res = lbfgsb(_loss, aug_grad, θ; m = cache.opt.m, pgtol = sqrt(ϵ), maxiter = 100, lb = cache.lb, ub = cache.ub)
            else
                res = lbfgsb(_loss, aug_grad, θ; m = cache.opt.m, pgtol = sqrt(ϵ), maxiter = 100)
            end
            @show res[2]
            @show res[1]
            @show cons_tmp
            @show λ
            @show β
            @show μ

            θ = res[2]
            cons_tmp .= 0.0
            cache.f.cons(cons_tmp, θ)
            λ = max.(min.(λmax , λ .+ ρ * cons_tmp[eq_inds]), λmin)
            β = max.(cons_tmp[ineq_inds], -1 .* μ ./ ρ)
            μ = min.(μmax, max.(μ .+ ρ * cons_tmp[ineq_inds], μmin))

            if max(norm(cons_tmp[eq_inds], Inf), norm(β, Inf)) > τ * max(norm(prev_eqcons, Inf), norm(prevβ, Inf))
                ρ = γ * ρ
            end
            if norm(cons_tmp[eq_inds], Inf) < ϵ && norm(β, Inf) < ϵ
                break
            end
        end

        stats = Optimization.OptimizationStats(; iterations = maxiters,
            time = 0.0, fevals = maxiters, gevals = maxiters)
        return SciMLBase.build_solution(cache, cache.opt, res[2], res[1], stats = stats)
    else
        _loss = function (θ)
            x = cache.f(θ, cache.p)

            opt_state = Optimization.OptimizationState(u = θ, objective = x[1])
            if cache.callback(opt_state, x...)
                error("Optimization halted by callback.")
            end
            return x[1]
        end

        t0 = time()
        if cache.lb !== nothing && cache.ub !== nothing
            res = lbfgsb(_loss, cache.f.grad, cache.u0; m = cache.opt.m, maxiter = maxiters,
                lb = cache.lb, ub = cache.ub)
        else
            res = lbfgsb(_loss, cache.f.grad, cache.u0; m = cache.opt.m, maxiter = maxiters)
        end

        t1 = time()
        stats = Optimization.OptimizationStats(; iterations = maxiters,
            time = t1 - t0, fevals = maxiters, gevals = maxiters)

        return SciMLBase.build_solution(cache, cache.opt, res[2], res[1], stats = stats)
    end
end
