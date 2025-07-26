module OptimizationNLopt

using Reexport
@reexport using NLopt, Optimization
using Optimization.SciMLBase
using Optimization: deduce_retcode

(f::NLopt.Algorithm)() = f

SciMLBase.allowsbounds(opt::Union{NLopt.Algorithm, NLopt.Opt}) = true
SciMLBase.supports_opt_cache_interface(opt::Union{NLopt.Algorithm, NLopt.Opt}) = true

function SciMLBase.requiresgradient(opt::Union{NLopt.Algorithm, NLopt.Opt})
    # https://github.com/JuliaOpt/NLopt.jl/blob/master/src/NLopt.jl#L18C7-L18C16
    str_opt = string(opt isa NLopt.Algorithm ? opt : opt.algorithm)
    return str_opt[2] != 'N'
end

#interferes with callback handling
# function SciMLBase.allowsfg(opt::Union{NLopt.Algorithm, NLopt.Opt})
#     str_opt = string(opt isa NLopt.Algorithm ? opt : opt.algorithm)
#     return str_opt[2] == 'D'
# end

function SciMLBase.requireshessian(opt::Union{NLopt.Algorithm, NLopt.Opt})
    # https://github.com/JuliaOpt/NLopt.jl/blob/master/src/NLopt.jl#L18C7-L18C16
    str_opt = string(opt isa NLopt.Algorithm ? opt : opt.algorithm)
    return !(str_opt[2] == 'N' || occursin(r"LD_LBFGS|LD_SLSQP", str_opt))
end

function SciMLBase.requiresconsjac(opt::Union{NLopt.Algorithm, NLopt.Opt})
    # https://github.com/JuliaOpt/NLopt.jl/blob/master/src/NLopt.jl#L18C7-L18C16
    str_opt = string(opt isa NLopt.Algorithm ? opt : opt.algorithm)
    return str_opt[3] ∈ ['O', 'I'] || str_opt[5] == 'G'
end

function SciMLBase.allowsconstraints(opt::NLopt.Algorithm)
    str_opt = string(opt)
    return occursin(r"AUGLAG|CCSA|MMA|COBYLA|ISRES|AGS|ORIG_DIRECT|SLSQP", str_opt)
end

function SciMLBase.requiresconsjac(opt::NLopt.Algorithm)
    str_opt = string(opt)
    return occursin(r"AUGLAG|CCSA|MMA|COBYLA|ISRES|AGS|ORIG_DIRECT|SLSQP", str_opt)
end

function SciMLBase.__init(prob::SciMLBase.OptimizationProblem, opt::NLopt.Algorithm,
        ; cons_tol = 1e-6,
        callback = (args...) -> (false),
        progress = false, kwargs...)
    return OptimizationCache(prob, opt; cons_tol, callback, progress,
        kwargs...)
end

function __map_optimizer_args!(cache::OptimizationCache, opt::NLopt.Opt;
        callback = nothing,
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        local_method::Union{NLopt.Algorithm, NLopt.Opt, Nothing} = nothing,
        local_maxiters::Union{Number, Nothing} = nothing,
        local_maxtime::Union{Number, Nothing} = nothing,
        local_options::Union{NamedTuple, Nothing} = nothing,
        kwargs...)
    if local_method !== nothing
        if isa(local_method, NLopt.Opt)
            if ndims(local_method) != length(cache.u0)
                error("Passed local NLopt.Opt optimization dimension does not match OptimizationProblem dimension.")
            end
            local_meth = local_method
        else
            local_meth = NLopt.Opt(local_method, length(cache.u0))
        end

        if !isnothing(local_options)
            for j in Dict(pairs(local_options))
                NLopt.nlopt_set_param(opt, j.first, j.second)
            end
        end

        if !(isnothing(local_maxiters))
            NLopt.maxeval!(local_meth, local_maxiters)
        end

        if !(isnothing(local_maxtime))
            NLopt.maxtime!(local_meth, local_maxtime)
        end

        NLopt.local_optimizer!(opt, local_meth)
    end

    # add optimiser options from kwargs
    for j in kwargs
        if j.first != :cons_tol
            NLopt.nlopt_set_param(opt, j.first, j.second)
        end
    end

    if cache.ub !== nothing
        opt.upper_bounds = cache.ub
    end

    if cache.lb !== nothing
        opt.lower_bounds = cache.lb
    end

    if !(isnothing(maxiters))
        NLopt.maxeval!(opt, maxiters)
    end

    if !(isnothing(maxtime))
        NLopt.maxtime!(opt, maxtime)
    end

    if !isnothing(abstol)
        NLopt.ftol_abs!(opt, abstol)
    end
    if !isnothing(reltol)
        NLopt.ftol_rel!(opt, reltol)
    end

    return nothing
end

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
        C
}) where {
        F,
        RC,
        LB,
        UB,
        LC,
        UC,
        S,
        O <:
        Union{
            NLopt.Algorithm,
            NLopt.Opt
        },
        D,
        P,
        C
}
    local x

    _loss = function (θ)
        x = cache.f(θ, cache.p)
        opt_state = Optimization.OptimizationState(u = θ, p = cache.p, objective = x[1])
        if cache.callback(opt_state, x...)
            NLopt.force_stop!(opt_setup)
        end
        return x[1]
    end

    fg! = function (θ, G)
        if length(G) > 0
            cache.f.grad(G, θ)
        end
        return _loss(θ)
    end

    opt_setup = if isa(cache.opt, NLopt.Opt)
        if ndims(cache.opt) != length(cache.u0)
            error("Passed NLopt.Opt optimization dimension does not match OptimizationProblem dimension.")
        end
        cache.opt
    else
        NLopt.Opt(cache.opt, length(cache.u0))
    end

    if cache.sense === Optimization.MaxSense
        NLopt.max_objective!(opt_setup, fg!)
    else
        NLopt.min_objective!(opt_setup, fg!)
    end

    if cache.f.cons !== nothing
        eqinds = map((y) -> y[1] == y[2], zip(cache.lcons, cache.ucons))
        ineqinds = map((y) -> y[1] != y[2], zip(cache.lcons, cache.ucons))
        cons_cache = zeros(eltype(cache.u0), sum(eqinds) + sum(ineqinds))
        thetacache = rand(size(cache.u0))
        Jthetacache = rand(size(cache.u0))
        Jcache = zeros(eltype(cache.u0), sum(ineqinds) + sum(eqinds), length(cache.u0))
        evalcons = function (θ, ineqoreq)
            if thetacache != θ
                cache.f.cons(cons_cache, θ)
                thetacache = copy(θ)
            end
            if ineqoreq == :eq
                return @view(cons_cache[eqinds])
            else
                return @view(cons_cache[ineqinds])
            end
        end

        evalconj = function (θ, ineqoreq)
            if Jthetacache != θ
                cache.f.cons_j(Jcache, θ)
                Jthetacache = copy(θ)
            end

            if ineqoreq == :eq
                return @view(Jcache[eqinds, :])'
            else
                return @view(Jcache[ineqinds, :])'
            end
        end

        if sum(ineqinds) > 0
            ineqcons = function (res, θ, J)
                res .= copy(evalcons(θ, :ineq))
                if length(J) > 0
                    J .= copy(evalconj(θ, :ineq))
                end
            end
            NLopt.inequality_constraint!(
                opt_setup, ineqcons, [cache.solver_args.cons_tol for i in 1:sum(ineqinds)])
        end
        if sum(eqinds) > 0
            eqcons = function (res, θ, J)
                res .= copy(evalcons(θ, :eq))
                if length(J) > 0
                    J .= copy(evalconj(θ, :eq))
                end
            end
            NLopt.equality_constraint!(
                opt_setup, eqcons, [cache.solver_args.cons_tol for i in 1:sum(eqinds)])
        end
    end

    maxiters = Optimization._check_and_convert_maxiters(cache.solver_args.maxiters)
    maxtime = Optimization._check_and_convert_maxtime(cache.solver_args.maxtime)

    __map_optimizer_args!(cache, opt_setup; callback = cache.callback, maxiters = maxiters,
        maxtime = maxtime,
        cache.solver_args...)

    t0 = time()
    (minf, minx, ret) = NLopt.optimize(opt_setup, cache.u0)
    t1 = time()
    retcode = deduce_retcode(ret)

    if retcode == ReturnCode.Failure
        @warn "NLopt failed to converge: $(ret)"
    end
    stats = Optimization.OptimizationStats(; time = t1 - t0)
    SciMLBase.build_solution(cache, cache.opt, minx,
        minf; original = opt_setup, retcode = retcode,
        stats = stats)
end

end
