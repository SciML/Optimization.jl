struct IpoptState
    alg_mod::Cint
    iter_count::Cint
    obj_value::Float64
    inf_pr::Float64
    inf_du::Float64
    mu::Float64
    d_norm::Float64
    regularization_size::Float64
    alpha_du::Float64
    alpha_pr::Float64
    ls_trials::Cint
    u::Vector{Float64}
    z_L::Vector{Float64}
    z_U::Vector{Float64}
    g::Vector{Float64}
    lambda::Vector{Float64}
end

struct IpoptProgressLogger{C, P}
    progress::Bool
    callback::C
    prob::P
    n::Int
    num_cons::Int
    maxiters::Union{Nothing, Int}
    iterations::Ref{Int}
    # caches for GetIpoptCurrentIterate
    u::Vector{Float64}
    z_L::Vector{Float64}
    z_U::Vector{Float64}
    g::Vector{Float64}
    lambda::Vector{Float64}
end

function IpoptProgressLogger(
        progress::Bool, callback::C, prob::P, n::Int, num_cons::Int,
        maxiters::Union{Nothing, Int}, iterations::Ref{Int}
    ) where {C, P}
    # Initialize caches
    u, z_L, z_U = zeros(n), zeros(n), zeros(n)
    g, lambda = zeros(num_cons), zeros(num_cons)
    return IpoptProgressLogger(
        progress, callback, prob, n, num_cons, maxiters, iterations, u, z_L, z_U, g, lambda
    )
end

function (cb::IpoptProgressLogger)(
        alg_mod::Cint,
        iter_count::Cint,
        obj_value::Float64,
        inf_pr::Float64,
        inf_du::Float64,
        mu::Float64,
        d_norm::Float64,
        regularization_size::Float64,
        alpha_du::Float64,
        alpha_pr::Float64,
        ls_trials::Cint
    )
    scaled = false
    Ipopt.GetIpoptCurrentIterate(
        cb.prob, scaled, cb.n, cb.u, cb.z_L, cb.z_U, cb.num_cons, cb.g, cb.lambda
    )

    original = IpoptState(
        alg_mod,
        iter_count,
        obj_value,
        inf_pr,
        inf_du,
        mu,
        d_norm,
        regularization_size,
        alpha_du,
        alpha_pr,
        ls_trials,
        cb.u,
        cb.z_L,
        cb.z_U,
        cb.g,
        cb.lambda
    )

    opt_state = OptimizationBase.OptimizationState(;
        iter = Int(iter_count), cb.u, objective = obj_value, original
    )
    cb.iterations[] = Int(iter_count)

    if cb.progress
        maxiters = cb.maxiters
        msg = "objective: " *
            sprint(show, obj_value, context = :compact => true)
        if !isnothing(maxiters)
            # we stop at either convergence or max_steps
            Base.@logmsg(
                Base.LogLevel(-1), msg, progress = iter_count / maxiters,
                _id = :OptimizationIpopt
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
