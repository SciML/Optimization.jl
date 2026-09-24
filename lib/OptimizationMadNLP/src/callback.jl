struct MadNLPProgressLogger{C, P} <: MadNLP.AbstractUserCallback
    callback::C
    progress::Bool
    maxiters::Union{Nothing, Int}
    p::P
end

function (cb::MadNLPProgressLogger)(solver::MadNLP.AbstractMadNLPSolver, mode)
    iter_count = solver.cnt.k
    obj_value = solver.obj_val
    u = solver.x

    opt_state = OptimizationBase.OptimizationState(;
        iter = iter_count, u, cb.p, objective = obj_value, original = solver
    )

    if cb.progress
        maxiters = cb.maxiters
        msg = "objective: " *
            sprint(show, obj_value, context = :compact => true)
        if !isnothing(maxiters)
            # we stop at either convergence or max_steps
            Base.@logmsg(
                Base.LogLevel(-1), msg, progress = iter_count / maxiters,
                _id = :OptimizationMadNLP
            )
        end
    end

    return if !isnothing(cb.callback)
        # return `true` to keep going, or `false` to terminate the optimization
        # this is the other way around compared to OptimizationBase.jl callbacks
        !cb.callback(opt_state, obj_value)
    else
        true
    end
end
