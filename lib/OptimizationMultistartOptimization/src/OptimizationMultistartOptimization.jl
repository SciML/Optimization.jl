module OptimizationMultistartOptimization

using Reexport
@reexport using MultistartOptimization, Optimization
using Optimization.SciMLBase

SciMLBase.requiresbounds(opt::MultistartOptimization.TikTak) = true
SciMLBase.allowsbounds(opt::MultistartOptimization.TikTak) = true
SciMLBase.supports_opt_cache_interface(opt::MultistartOptimization.TikTak) = true

function SciMLBase.__init(prob::SciMLBase.OptimizationProblem, opt::MultistartOptimization.TikTak,
                          local_opt,
                          data = Optimization.DEFAULT_DATA;
                          kwargs...)
    return OptimizationCache(prob, opt, data; local_opt = local_opt,
                             kwargs...)
end

function SciMLBase.__solve(cache::OptimizationCache)
    local x, _loss

    _loss = function (θ)
        x = cache.f(θ, cache.p)
        return first(x)
    end

    opt_setup = MultistartOptimization.MinimizationProblem(_loss, cache.lb, cache.ub)

    _local_optimiser = function (pb, θ0, prob)
        prob_tmp = remake(prob, u0 = θ0)
        res = SciMLBase.__solve(prob_tmp, cache.solver_args.local_opt;
                    kwargs...)
        return (value = res.minimum, location = res.minimizer, ret = res.retcode)
    end

    local_optimiser(pb, θ0) = _local_optimiser(pb, θ0, prob)

    t0 = time()
    opt_res = MultistartOptimization.multistart_minimization(cache.opt, local_optimiser,
                                                             opt_setup;
                                                             use_threads = use_threads)
    t1 = time()
    opt_ret = hasproperty(opt_res, :ret) ? opt_res.ret : nothing

    SciMLBase.build_solution(cache,
                             (cache.opt, cache.solver_args.local_opt), opt_res.location,
                             opt_res.value;
                             (isnothing(opt_ret) ? (; original = opt_res) :
                              (; original = opt_res, retcode = opt_ret,
                               solve_time = t1 - t0))...)
end

end
