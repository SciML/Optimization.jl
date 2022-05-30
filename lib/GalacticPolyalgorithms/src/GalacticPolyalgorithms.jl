module GalacticPolyalgorithms

using GalacticOptim, GalacticOptim.SciMLBase, GalacticOptimJL, GalacticOptimisers

struct PolyOpt end

function SciMLBase.__solve(prob::OptimizationProblem,
                           opt::PolyOpt,
                           args...;
                           maxiters = nothing,
                           kwargs...)

    loss, θ = x -> prob.f(x, prob.p), prob.u0
    deterministic = first(loss(θ)) == first(loss(θ))

    if (!isempty(args) || !deterministic) && maxiters === nothing
        error("Automatic optimizer determination requires deterministic loss functions (and no data) or maxiters must be specified.")
    end

    if isempty(args) && deterministic && prob.lb === nothing && prob.ub === nothing
        # If determinsitic then ADAM -> finish with BFGS
        if maxiters === nothing
            res1 = GalacticOptim.solve(prob, Optimisers.ADAM(0.01), args...; maxiters=300, kwargs...)
        else
            res1 = GalacticOptim.solve(prob, Optimisers.ADAM(0.01), args...; maxiters, kwargs...)
        end

        optprob2 = GalacticOptim.OptimizationProblem(
            prob.f, res1.u; lb=prob.lb, ub=prob.ub, kwargs...)
        res1 = GalacticOptim.solve(
            optprob2, BFGS(initial_stepnorm=0.01), args...; maxiters, kwargs...)
    elseif isempty(args) && deterministic
        res1 = GalacticOptim.solve(
            prob, BFGS(initial_stepnorm=0.01), args...; maxiters, kwargs...)
    else
        res1 = GalacticOptim.solve(prob, Optimisers.ADAM(0.1), args...; maxiters, kwargs...)
    end

end

export PolyOpt

end