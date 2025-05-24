module OptimizationODE

using Reexport
@reexport using Optimization
using Optimization.SciMLBase

export ODEGradientDescent

# The optimizer “type”

struct ODEGradientDescent end

# capability flags
SciMLBase.requiresbounds(::ODEGradientDescent)           = false
SciMLBase.allowsbounds(::ODEGradientDescent)             = false
SciMLBase.allowscallback(::ODEGradientDescent)           = false
SciMLBase.supports_opt_cache_interface(::ODEGradientDescent) = true
SciMLBase.requiresgradient(::ODEGradientDescent)         = true
SciMLBase.requireshessian(::ODEGradientDescent)          = false
SciMLBase.requiresconsjac(::ODEGradientDescent)          = false
SciMLBase.requiresconshess(::ODEGradientDescent)         = false

# Map standard kwargs to our solver’s args

function __map_optimizer_args!(
    cache::OptimizationCache, opt::ODEGradientDescent;
    callback    = nothing,
    maxiters::Union{Number,Nothing}=nothing,
    maxtime::Union{Number,Nothing}=nothing,
    abstol::Union{Number,Nothing}=nothing,
    reltol::Union{Number,Nothing}=nothing,
    η::Float64 = 0.1,
    tmax::Float64 = 1.0,
    dt::Float64 = 0.01,
    kwargs...
)
    # override our defaults
    cache.solver_args = merge(cache.solver_args, (
        η = η,
        tmax = tmax,
        dt = dt,
    ))
    # now apply common options
    if !(isnothing(maxiters))
        cache.solver_args.maxiters = maxiters
    end
    if !(isnothing(maxtime))
        cache.solver_args.maxtime = maxtime
    end
    return nothing
end

# 3) Initialize the cache (captures f, u0, bounds, and solver_args)

function SciMLBase.__init(
    prob::SciMLBase.OptimizationProblem,
    opt::ODEGradientDescent,
    data = Optimization.DEFAULT_DATA;
    η::Float64 = 0.1,
    tmax::Float64 = 1.0,
    dt::Float64 = 0.01,
    callback  = (args...)->false,
    progress  = false,
    kwargs...
)
    return OptimizationCache(
        prob, opt, data;
        η        = η,
        tmax     = tmax,
        dt       = dt,
        callback = callback,
        progress = progress,
        maxiters = nothing,
        maxtime  = nothing,
        kwargs...
    )
end

# 4) The actual solve loop: Euler integration of gradient descent

function SciMLBase.__solve(
    cache::OptimizationCache{F,RC,LB,UB,LC,UC,S,O,D,P,C}
) where {F,RC,LB,UB,LC,UC,S,O<:ODEGradientDescent,D,P,C}

    # unpack initial state & parameters
    u0      = cache.u0
    η       = get(cache.solver_args, :η, 0.1)
    tmax    = get(cache.solver_args, :tmax, 1.0)
    dt      = get(cache.solver_args, :dt, 0.01)
    maxiter = get(cache.solver_args, :maxiters, nothing)

    # prepare working storage
    u = copy(u0)
    G = similar(u)

    t    = 0.0
    iter = 0
    # Euler loop
    while (isnothing(maxiter) || iter < maxiter) && t <= tmax
        # compute gradient in‐place
        cache.f.grad(G, u, cache.p)
        # Euler step
        u .-= η .* G
        t    += dt
        iter += 1
    end

    # final objective
    fval = cache.f(u, cache.p)

    # record stats: one final f‐eval, iter gradient‐evals
    stats = Optimization.OptimizationStats(
        iterations = iter,
        time       = 0.0,        # could time() if you like
        fevals     = 1,
        gevals     = iter,
        hevals     = 0
    )

    return SciMLBase.build_solution(
        cache, cache.opt,
        u,
        fval,
        retcode = ReturnCode.Success,
        stats   = stats
    )
end

end # module
