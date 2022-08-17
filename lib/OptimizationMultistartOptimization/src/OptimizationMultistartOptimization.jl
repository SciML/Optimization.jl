module OptimizationMultistartOptimization

using Reexport, Optimization, Optimization.SciMLBase
@reexport using MultistartOptimization

function SciMLBase.__solve(prob::OptimizationProblem,
                           multiopt::MultistartOptimization.TikTak, opt;
                           use_threads = true,
                           kwargs...)
    local x, _loss

    _loss = function (θ)
        x = prob.f(θ, prob.p)
        return first(x)
    end

    opt_setup = MultistartOptimization.MinimizationProblem(_loss, prob.lb, prob.ub)

    _local_optimiser = function (pb, θ0, prob)
        prob_tmp = remake(prob, u0 = θ0)
        res = solve(prob_tmp, opt;
                    kwargs...)
        return (value = res.minimum, location = res.minimizer, ret = res.retcode)
    end

    local_optimiser(pb, θ0) = _local_optimiser(pb, θ0, prob)

    t0 = time()
    opt_res = MultistartOptimization.multistart_minimization(multiopt, local_optimiser,
                                                             opt_setup;
                                                             use_threads = use_threads)
    t1 = time()
    opt_ret = hasproperty(opt_res, :ret) ? opt_res.ret : nothing

    SciMLBase.build_solution(prob, (multiopt, opt), opt_res.location, opt_res.value;
                             (isnothing(opt_ret) ? (; original = opt_res) :
                              (; original = opt_res, retcode = opt_ret))...)
end

end
