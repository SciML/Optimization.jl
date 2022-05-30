module GalacticPolyalgorithms

using GalacticOptim, GalacticOptim.SciMLBase, GalacticOptimJL, GalacticOptimisers

struct PolyOpt end

function SciMLBase.__solve(prob::OptimizationProblem,
                           opt::PolyOpt,
                           args...;
                           maxiters = nothing,
                           kwargs...)

    loss, θ = prob.f, prob.u0
    deterministic = first(loss(θ)) == first(loss(θ))

    if (!isempty(args) || !deterministic) && maxiters === nothing
        error("Automatic optimizer determination requires deterministic loss functions (and no data) or maxiters must be specified.")
    end

    if isempty(args) && deterministic && lower_bounds === nothing && upper_bounds === nothing
        # If determinsitic then ADAM -> finish with BFGS
        if maxiters === nothing
            res1 = GalacticOptim.solve(optprob, Optimisers.ADAM(0.01), args...; maxiters=300, kwargs...)
        else
            res1 = GalacticOptim.solve(optprob, Optimisers.ADAM(0.01), args...; maxiters, kwargs...)
        end

        optprob2 = GalacticOptim.OptimizationProblem(
            optfunc, res1.u; lb=lower_bounds, ub=upper_bounds, kwargs...)
        res1 = GalacticOptim.solve(
            optprob2, BFGS(initial_stepnorm=0.01), args...; maxiters, kwargs...)
    elseif isempty(args) && deterministic
        res1 = GalacticOptim.solve(
            optprob, BFGS(initial_stepnorm=0.01), args...; maxiters, kwargs...)
    else
        res1 = GalacticOptim.solve(optprob, Optimisers.ADAM(0.1), args...; maxiters, kwargs...)
    end

end

export PolyOpt

end