module OptimizationMultistartOptimization

using Reexport
@reexport using MultistartOptimization, Optimization
using Optimization.SciMLBase

SciMLBase.requiresbounds(opt::MultistartOptimization.TikTak) = true
SciMLBase.allowsbounds(opt::MultistartOptimization.TikTak) = true
SciMLBase.allowscallback(opt::MultistartOptimization.TikTak) = false
SciMLBase.supports_opt_cache_interface(opt::MultistartOptimization.TikTak) = true

function SciMLBase.__init(prob::SciMLBase.OptimizationProblem,
        opt::MultistartOptimization.TikTak,
        local_opt,
        data = Optimization.DEFAULT_DATA;
        use_threads = true,
        kwargs...)
    return OptimizationCache(prob, opt, data; local_opt = local_opt, prob = prob,
        use_threads = use_threads,
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
        MultistartOptimization.TikTak,
        D,
        P,
        C
}
    local x, _loss

    _loss = function (θ)
        x = cache.f(θ, cache.p)
        return first(x)
    end

    opt_setup = MultistartOptimization.MinimizationProblem(_loss, cache.lb, cache.ub)

    _local_optimiser = function (pb, θ0, prob)
        prob_tmp = remake(prob, u0 = θ0)
        res = solve(prob_tmp, cache.solver_args.local_opt)
        return (value = res.minimum, location = res.minimizer, ret = res.retcode)
    end

    local_optimiser(pb, θ0) = _local_optimiser(pb, θ0, cache.solver_args.prob)

    t0 = time()
    opt_res = MultistartOptimization.multistart_minimization(cache.opt, local_optimiser,
        opt_setup;
        use_threads = cache.solver_args.use_threads)
    t1 = time()
    opt_ret = hasproperty(opt_res, :ret) ? opt_res.ret : nothing
    stats = Optimization.OptimizationStats(; time = t1 - t0)
    SciMLBase.build_solution(cache,
        (cache.opt, cache.solver_args.local_opt), opt_res.location,
        opt_res.value;
        stats = stats,
        (isnothing(opt_ret) ? (; original = opt_res) :
         (; original = opt_res, retcode = opt_ret))...)
end

end
