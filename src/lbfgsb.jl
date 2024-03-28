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
