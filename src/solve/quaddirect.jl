export QuadDirect

struct QuadDirect end

function __solve(prob::OptimizationProblem, opt::QuadDirect; splits = nothing, maxiters = nothing, kwargs...)

    local x, _loss

    if !(isnothing(maxiters)) && maxiters <= 0.0
        error("The number of maxiters has to be a non-negative and non-zero number.")
    elseif !(isnothing(maxiters))
        maxiters = convert(Int, maxiters)
    end

    if splits === nothing
        error("You must provide the initial locations at which to evaluate the function in `splits` (a list of 3-vectors with values in strictly increasing order and within the specified bounds).")
    end

    _loss = function(θ)
        x = prob.f(θ, prob.p)
        return first(x)
    end

    t0 = time()

    root, x0 = !(isnothing(maxiters)) ? QuadDIRECT.analyze(_loss, splits, prob.lb, prob.ub; maxevals = maxiters, kwargs...) : QuadDIRECT.analyze(_loss, splits, prob.lb, prob.ub; kwargs...)
    box = minimum(root)
    t1 = time()

    SciMLBase.build_solution(prob, opt, QuadDIRECT.position(box, x0), QuadDIRECT.value(box); original=root)
end
