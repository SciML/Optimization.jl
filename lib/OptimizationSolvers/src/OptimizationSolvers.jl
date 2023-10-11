module OptimizationSolvers

using Reexport, Printf, ProgressLogging
@reexport using Optimization
using Optimization.SciMLBase

SciMLBase.supports_opt_cache_interface(opt::AbstractRule) = true
include("sophia.jl")

function SciMLBase.__init(prob::SciMLBase.OptimizationProblem, opt::AbstractRule,
    data = Optimization.DEFAULT_DATA; save_best = true,
    callback = (args...) -> (false),
    progress = false, kwargs...)
    return OptimizationCache(prob, opt, data; save_best, callback, progress,
        kwargs...)
end

struct BFGS end

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
    C,
}) where {
    F,
    RC,
    LB,
    UB,
    LC,
    UC,
    S,
    O <:BFGS,
    D,
    P,
    C,
}
    if cache.data != Optimization.DEFAULT_DATA
        maxiters = length(cache.data)
        data = cache.data
    else
        maxiters = Optimization._check_and_convert_maxiters(cache.solver_args.maxiters)
        data = Optimization.take(cache.data, maxiters)
    end
    opt = cache.opt
    θ = copy(cache.u0)
    G = copy(θ)

    H = I(length(θ)) * γ

    t0 = time()
    Optimization.@withprogress cache.progress name="Training" begin
        for (i, d) in enumerate(data)
            #bfgs
            
        end
    end

    t1 = time()

    SciMLBase.build_solution(cache, cache.opt, θ, x[1], solve_time = t1 - t0)
    # here should be build_solution to create the output message
end

end
