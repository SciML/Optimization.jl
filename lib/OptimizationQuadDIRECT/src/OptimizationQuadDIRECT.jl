module OptimizationQuadDIRECT

using Reexport
@reexport using Optimization
using QuadDIRECT, Optimization.SciMLBase

export QuadDirect

struct QuadDirect end

SciMLBase.allowsbounds(::QuadDirect) = true
SciMLBase.requiresbounds(::QuadDirect) = true
SciMLBase.allowscallback(::QuadDirect) = false

function __map_optimizer_args(prob::OptimizationProblem, opt::QuadDirect;
                              callback = nothing,
                              maxiters::Union{Number, Nothing} = nothing,
                              maxtime::Union{Number, Nothing} = nothing,
                              abstol::Union{Number, Nothing} = nothing,
                              reltol::Union{Number, Nothing} = nothing)
    if !isnothing(maxtime)
        @warn "common maxtime is currently not used by $(opt)"
    end

    mapped_args = (;)

    if !isnothing(maxiters)
        mapped_args = (; mapped_args..., maxevals = maxiters)
    end

    if !isnothing(abstol)
        mapped_args = (; mapped_args..., atol = abstol)
    end

    if !isnothing(reltol)
        mapped_args = (; mapped_args..., rtol = reltol)
    end

    return mapped_args
end

function SciMLBase.__solve(prob::OptimizationProblem, opt::QuadDirect;
                           splits = nothing,
                           maxiters::Union{Number, Nothing} = nothing,
                           maxtime::Union{Number, Nothing} = nothing,
                           abstol::Union{Number, Nothing} = nothing,
                           reltol::Union{Number, Nothing} = nothing,
                           kwargs...)
    local x, _loss

    maxiters = Optimization._check_and_convert_maxiters(maxiters)

    if splits === nothing
        error("You must provide the initial locations at which to evaluate the function in `splits` (a list of 3-vectors with values in strictly increasing order and within the specified bounds).")
    end

    _loss = function (θ)
        x = prob.f(θ, prob.p)
        return first(x)
    end

    opt_arg = __map_optimizer_args(prob, opt; maxiters = maxiters, maxtime = maxtime,
                                   abstol = abstol, reltol = reltol, kwargs...)
    t0 = time()
    # root, x0 = !(isnothing(maxiters)) ? QuadDIRECT.analyze(_loss, splits, prob.lb, prob.ub; maxevals = maxiters, kwargs...) : QuadDIRECT.analyze(_loss, splits, prob.lb, prob.ub; kwargs...)
    root, x0 = QuadDIRECT.analyze(_loss, splits, prob.lb, prob.ub; opt_arg...)
    box = minimum(root)
    t1 = time()

    SciMLBase.build_solution(SciMLBase.DefaultOptimizationCache(prob.f, prob.p), opt,
                             QuadDIRECT.position(box, x0), QuadDIRECT.value(box);
                             original = root, solve_time = t1 - t0)
end

end
