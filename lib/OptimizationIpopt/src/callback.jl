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
    z_L::Vector{Float64}
    z_U::Vector{Float64}
    lambda::Vector{Float64}
end

struct IpoptProgressLogger{C <: IpoptCache, P}
    progress::Bool
    cache::C
    prob::P
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
    n = cb.cache.n
    m = cb.cache.num_cons
    u, z_L, z_U = zeros(n), zeros(n), zeros(n)
    g, lambda = zeros(m), zeros(m)
    scaled = false
    Ipopt.GetIpoptCurrentIterate(cb.prob, scaled, n, u, z_L, z_U, m, g, lambda)

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
        z_L,
        z_U,
        lambda
    )

    opt_state = Optimization.OptimizationState(;
        iter = Int(iter_count), u, objective = obj_value, original)
    cb.cache.iterations = iter_count

    if cb.cache.progress
        maxiters = cb.cache.solver_args.maxiters
        msg = "objective: " *
              sprint(show, obj_value, context = :compact => true)
        if !isnothing(maxiters)
            # we stop at either convergence or max_steps
            Base.@logmsg(Base.LogLevel(-1), msg, progress=iter_count / maxiters,
                _id=:OptimizationIpopt)
        end
    end
    if !isnothing(cb.cache.callback)
        # return `true` to keep going, or `false` to terminate the optimization
        # this is the other way around compared to Optimization.jl callbacks
        !cb.cache.callback(opt_state, obj_value)
    else
        true
    end
end
